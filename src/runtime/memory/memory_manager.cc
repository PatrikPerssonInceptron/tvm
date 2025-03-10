/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/runtime/memory/memory_manager.cc
 * \brief Allocate and manage memory for the runtime.
 */
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/registry.h>

#include <memory>
#include <utility>

#include "naive_allocator.h"
#include "pooled_allocator.h"

namespace tvm {
namespace runtime {
namespace memory {

static void BufferDeleter(Object* obj) {
  auto* ptr = static_cast<NDArray::Container*>(obj);
  ICHECK(ptr->manager_ctx != nullptr);
  Buffer* buffer = reinterpret_cast<Buffer*>(ptr->manager_ctx);
  MemoryManager::GetAllocator(buffer->device, buffer->alloc_type)->Free(*(buffer));
  delete buffer;
  delete ptr;
}

Storage::Storage(Buffer buffer, Allocator* allocator) {
  auto n = make_object<StorageObj>();
  n->buffer = std::move(buffer);
  n->allocator = allocator;
  data_ = std::move(n);
}

void StorageObj::Deleter(Object* obj) {
  auto* ptr = static_cast<NDArray::Container*>(obj);
  // When invoking AllocNDArray we don't own the underlying allocation
  // and should not delete the buffer, but instead let it be reclaimed
  // by the storage object's destructor.
  //
  // We did bump the reference count by 1 to keep alive the StorageObj
  // allocation in case this NDArray is the sole owner.
  //
  // We decrement the object allowing for the buffer to release our
  // reference count from allocation.
  StorageObj* storage = reinterpret_cast<StorageObj*>(ptr->manager_ctx);
  storage->DecRef();
  delete ptr;
}

inline void VerifyDataType(DLDataType dtype) {
  ICHECK_GE(dtype.lanes, 1);
  if (dtype.code == kDLFloat) {
    ICHECK_EQ(dtype.bits % 8, 0);
  } else {
    // allow uint1 as a special flag for bool.
    if (dtype.bits == 1 && dtype.code == kDLUInt) return;
    ICHECK_EQ(dtype.bits % 8, 0);
  }
  ICHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataAlignment(const DLTensor& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

void StorageObj::ScopedDeleter(Object* obj) {
  auto* ptr = static_cast<NDArray::Container*>(obj);
  StorageObj* storage = reinterpret_cast<StorageObj*>(ptr->manager_ctx);

  // Let the device handle proper cleanup of view
  storage->allocator->FreeView(ptr->dl_tensor.device, ptr->dl_tensor.data);
  storage->DecRef();
  delete ptr;
}

NDArray StorageObj::AllocNDArrayScoped(int64_t offset, ShapeTuple shape, DLDataType dtype,
                                       String scope) {
  if (scope == "global" || scope.empty()) {
    return AllocNDArray(offset, shape, dtype);
  }
  VerifyDataType(dtype);
  void* data = this->allocator->CreateView(this->buffer, shape, dtype, scope);
  NDArray::Container* container = new NDArray::Container(data, shape, dtype, this->buffer.device);
  container->dl_tensor.byte_offset = offset;
  container->SetDeleter(StorageObj::ScopedDeleter);
  size_t needed_size = DeviceAPI::Get(this->buffer.device)->GetDataSize(container->dl_tensor);
  this->IncRef();
  container->manager_ctx = reinterpret_cast<void*>(this);
  NDArray ret(GetObjectPtr<Object>(container));
  // RAII in effect, now run the check.
  ICHECK(offset + needed_size <= this->buffer.size)
      << "storage allocation failure, attempted to allocate " << needed_size << " at offset "
      << offset << " in region that is " << this->buffer.size << "bytes";
  return ret;
}

NDArray StorageObj::AllocNDArray(int64_t offset, ShapeTuple shape, DLDataType dtype) {
  VerifyDataType(dtype);

  // crtical zone: allocate header, cannot throw
  NDArray::Container* container =
      new NDArray::Container(this->buffer.data, shape, dtype, this->buffer.device);
  container->dl_tensor.byte_offset = offset;

  container->SetDeleter(StorageObj::Deleter);
  size_t needed_size = DeviceAPI::Get(this->buffer.device)->GetDataSize(container->dl_tensor);
  this->IncRef();
  // The manager context pointer must continue to point to the storage object
  // which owns the backing memory, and keeps track of the reference count.
  //
  // When we free a container we extract the storage object, decrement its
  // reference count, then destroy the container, but leave the underlying
  // buffer intact.
  container->manager_ctx = reinterpret_cast<void*>(this);

  if (this->buffer.device.device_type == kDLHexagon) {
    // For Hexagon, non-zero offset support simply requires adjusting the
    // beginning of data pointer
    auto offset_ptr = reinterpret_cast<uint8_t*>(this->buffer.data) + offset;
    container->dl_tensor.data = reinterpret_cast<void*>(offset_ptr);
    container->dl_tensor.byte_offset = 0;
  }

  NDArray ret(GetObjectPtr<Object>(container));
  // RAII in effect, now run the check.

  ICHECK(offset + needed_size <= this->buffer.size)
      << "storage allocation failure, attempted to allocate " << needed_size << " at offset "
      << offset << " in region that is " << this->buffer.size << "bytes";

  return ret;
}

MemoryManager* MemoryManager::Global() {
  // NOTE: explicitly use new to avoid exit-time destruction of global state
  // Global state will be recycled by OS as the process exits.
  static auto* inst = new MemoryManager();
  return inst;
}

std::string DeviceTypeStr(DLDeviceType type) {
  switch (type) {
    case kDLOpenCL:
      return "opencl";
      break;
    case kDLVulkan:
      return "vulkan";
      break;
    default:
      return "";
  }
}

Allocator* GetDeviceSpecificAllocator(Device dev, AllocatorType type) {
  std::string dev_str = DeviceTypeStr(dev.device_type);
  auto* device_alloc_helper = tvm::runtime::Registry::Get("DeviceAllocator." + dev_str);
  void* valloc;
  Allocator* allocator = nullptr;
  if (device_alloc_helper) {
    valloc = (*device_alloc_helper)(dev, static_cast<int>(type));
    allocator = static_cast<Allocator*>(valloc);
  }
  if (nullptr == allocator) {
    switch (type) {
      case kNaive: {
        VLOG(1) << "New naive allocator for " << dev;
        allocator = new NaiveAllocator();
        break;
      }
      case kPooled: {
        VLOG(1) << "New pooled allocator for " << dev;
        allocator = new PooledAllocator();
        break;
      }
      default:
        LOG(FATAL) << "Unknown allocator type: " << type;
    }
  }
  return allocator;
}

Allocator* MemoryManager::GetOrCreateAllocator(Device dev, AllocatorType type) {
  MemoryManager* m = MemoryManager::Global();
  std::lock_guard<std::mutex> lock(m->mu_);
  if (m->allocators_.find(dev) == m->allocators_.end()) {
    m->allocators_.emplace(dev, std::unordered_map<AllocatorType, std::unique_ptr<Allocator>>());
  }
  if (m->allocators_.at(dev).find(type) == m->allocators_.at(dev).end()) {
    std::unique_ptr<Allocator> alloc;
    alloc.reset(GetDeviceSpecificAllocator(dev, type));
    auto ret = alloc.get();
    m->allocators_.at(dev).emplace(type, std::move(alloc));
    return ret;
  }
  auto alloc = m->allocators_.at(dev).at(type).get();

  return alloc;
}

Allocator* MemoryManager::GetAllocator(Device dev, AllocatorType type) {
  MemoryManager* m = MemoryManager::Global();
  std::lock_guard<std::mutex> lock(m->mu_);
  auto it = m->allocators_.find(dev);
  if (it == m->allocators_.end()) {
    LOG(FATAL) << "Allocator for " << dev << " has not been created yet.";
  }
  if (it->second.find(type) == it->second.end()) {
    LOG(FATAL) << "Allocator for " << dev << " of type " << type << " has not been created yet.";
  }
  return it->second.at(type).get();
}

void MemoryManager::Clear() {
  MemoryManager* m = MemoryManager::Global();
  std::lock_guard<std::mutex> lock(m->mu_);
  for (const auto& [device, allocators] : m->allocators_) {
    for (const auto& [allocator_type, allocator] : allocators) {
      allocator->Clear();
    }
  }
}

NDArray Allocator::Empty(ShapeTuple shape, DLDataType dtype, DLDevice dev,
                         Optional<String> mem_scope) {
  VerifyDataType(dtype);
  NDArray::Container* container = new NDArray::Container(nullptr, shape, dtype, dev);
  container->SetDeleter(BufferDeleter);
  size_t size = DeviceAPI::Get(dev)->GetDataSize(container->dl_tensor, mem_scope);
  size_t alignment = GetDataAlignment(container->dl_tensor);
  Buffer* buffer = new Buffer;
  if (!mem_scope.defined() || mem_scope.value().empty() || mem_scope.value() == "global") {
    *buffer = this->Alloc(dev, size, alignment, dtype);
  } else {
    *buffer = this->Alloc(dev, shape, dtype, mem_scope.value());
  }
  container->manager_ctx = reinterpret_cast<void*>(buffer);
  container->dl_tensor.data = buffer->data;
  return NDArray(GetObjectPtr<Object>(container));
}

bool Allocator::AllowMemoryScope(const std::string& mem_scope) const {
  return mem_scope.empty() || mem_scope == "global";
}

Buffer Allocator::Alloc(Device dev, ShapeTuple shape, DLDataType type_hint,
                        const std::string& mem_scope) {
  if (AllowMemoryScope(mem_scope)) {
    // by default, we can always redirect to the flat memory allocations
    NDArray::Container container(nullptr, shape, type_hint, dev);
    size_t size = DeviceAPI::Get(dev)->GetDataSize(container.dl_tensor);
    size_t alignment = GetDataAlignment(container.dl_tensor);
    return Alloc(dev, size, alignment, type_hint);
  }
  LOG(FATAL) << "Allocator cannot allocate data space with "
             << "specified memory scope: " << mem_scope;
  return {};
}

void Allocator::Clear() {
  // This function by default does nothing.
  // For naive allocator, no explicit manual clear is needed.
  // Pooled allocator will override this method.
}

TVM_REGISTER_GLOBAL("vm.builtin.memory_manager.clear").set_body_typed(MemoryManager::Clear);

}  // namespace memory
}  // namespace runtime
}  // namespace tvm
