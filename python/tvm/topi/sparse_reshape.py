# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks
"""Sparse_Reshape operator"""
from ..tir import decl_buffer, ir_builder, Cast
from ..te import extern, div, floordiv, floormod


def sparse_reshape(
    sparse_indices,
    prev_shape,
    new_shape,
    new_sparse_indices_shape,
    new_shape_shape,
):
    """
    Reshape a Sparse Tensor

    Parameters
    ----------
    sparse_indices : te.Expr
        A 2-D tensor[N, n_dim] of integers containing location of sparse values, where N is the
        number of sparse values and n_dim is the number of dimensions of the dense_shape

    prev_shape : te.Expr
        A 1-D tensor containing the previous shape of the dense tensor

    new_shape : te.Expr
        A 1-D tensor containing the new shape of the dense tensor

    Returns
    -------
    result: te.Expr
        Output tensor.

    Examples
    --------
    .. code-block:: python

        sparse_indices = [[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            [1, 2, 3]]
        prev_shape = [2, 3, 4]
        new_shape = [9, -1]
        new_sparse_indices, new_shape = topi.sparse_reshape(
            sparse_indices, prev_shape, new_shape)
        new_sparse_indices = [[0, 0],
                              [0, 1],
                              [1, 2],
                              [4, 2],
                              [8, 1]]
        new_shape = [9, 4]
    """

    def gen_ir(
        sparse_indices_ptr,
        prev_shape_ptr,
        new_shape_ptr,
        new_sparse_indices_ptr,
        out_new_shape_ptr,
    ):
        ib = ir_builder.create()

        sparse_indices = ib.buffer_ptr(sparse_indices_ptr)
        prev_shape = ib.buffer_ptr(prev_shape_ptr)

        new_shape = ib.buffer_ptr(new_shape_ptr)
        out_new_shape = ib.buffer_ptr(out_new_shape_ptr)
        new_sparse_indices = ib.buffer_ptr(new_sparse_indices_ptr)
        out_new_shape = ib.buffer_ptr(out_new_shape_ptr)

        prev_shape_size = prev_shape_ptr.shape[0]
        new_shape_size = new_shape_ptr.shape[0]

        multipliers = ib.allocate(
            new_shape_ptr.dtype, (prev_shape_size,), name="multipliers", scope="local"
        )
        dividers = ib.allocate(
            new_shape_ptr.dtype, (new_shape_size,), name="dividers", scope="local"
        )
        flattened_indices = ib.allocate(
            new_shape_ptr.dtype,
            (sparse_indices_ptr.shape[0],),
            name="flattened_indices",
            scope="local",
        )

        total_ele = ib.allocate(new_shape_ptr.dtype, (1,), name="total_ele", scope="local")
        total_ele[0] = prev_shape[0]

        # Cumulative Reverse Exclusive Multiply
        multipliers[prev_shape_size - 1] = Cast(new_shape_ptr.dtype, 1)
        with ib.for_range(0, prev_shape_size - 1) as i_:
            i = i_ + 1
            multipliers[prev_shape_size - 1 - i] = (
                prev_shape[prev_shape_size - i] * multipliers[prev_shape_size - i]
            )
            total_ele[0] *= prev_shape[prev_shape_size - i]

        division_total_ele = ib.allocate(
            new_shape_ptr.dtype, (1,), name="division_total_ele", scope="local"
        )
        division_total_ele[0] = Cast(new_shape_ptr.dtype, 1)
        with ib.for_range(0, new_shape_size) as i:
            with ib.if_scope(new_shape[i] != -1):
                division_total_ele[0] *= new_shape[i]

        # Compute true output shape (replace negative ones)
        with ib.for_range(0, new_shape_size) as i:
            with ib.if_scope(new_shape[i] == -1):
                out_new_shape[i] = Cast(
                    new_shape_ptr.dtype, div(total_ele[0], division_total_ele[0])
                )
            with ib.else_scope():
                out_new_shape[i] = new_shape[i]

        equal_shape = ib.allocate("bool", (1,), name="equal_shape", scope="local")

        # Check if prev_shape and new_shape are equal
        equal_shape[0] = True
        with ib.if_scope(prev_shape_size == new_shape_size):
            with ib.for_range(0, prev_shape_size) as i:
                with ib.if_scope(prev_shape[i] != out_new_shape[i]):
                    equal_shape[0] = False
        with ib.else_scope():
            equal_shape[0] = False

        # Return same inputs if shapes are equal
        with ib.if_scope(equal_shape[0]):
            with ib.for_range(0, sparse_indices_ptr.shape[0], kind="parallel") as i:
                with ib.for_range(0, sparse_indices_ptr.shape[1]) as j:
                    new_sparse_indices[i, j] = sparse_indices[i, j]

        # Else compute new_sparse_indices
        with ib.else_scope():
            dividers[new_shape_size - 1] = Cast(new_shape_ptr.dtype, 1)
            with ib.for_range(0, new_shape_size - 1) as i_:
                i = i_ + 1
                dividers[new_shape_size - 1 - i] = (
                    dividers[new_shape_size - i] * out_new_shape[new_shape_size - i]
                )

            with ib.for_range(0, sparse_indices_ptr.shape[0], kind="parallel") as i:
                flattened_indices[i] = Cast(new_shape_ptr.dtype, 0)
                with ib.for_range(0, sparse_indices_ptr.shape[1]) as j:
                    flattened_indices[i] += sparse_indices[i, j] * multipliers[j]

            with ib.for_range(0, new_sparse_indices_ptr.shape[0], kind="parallel") as i:
                current_element = ib.allocate(
                    new_shape_ptr.dtype, (1,), name="current_element", scope="local"
                )
                current_element[0] = flattened_indices[i]

                with ib.for_range(0, new_sparse_indices_ptr.shape[1]) as j:
                    new_sparse_indices[i, j] = Cast(
                        sparse_indices_ptr.dtype, floordiv(current_element[0], dividers[j])
                    )
                    current_element[0] = floormod(current_element[0], dividers[j])

        return ib.get()

    new_sparse_indices_buf = decl_buffer(
        new_sparse_indices_shape, sparse_indices.dtype, "new_sparse_indices_buf"
    )
    new_shape_buf = decl_buffer(new_shape_shape, prev_shape.dtype, "new_shape_buf")

    return extern(
        [new_sparse_indices_shape, new_shape_shape],
        [sparse_indices, prev_shape, new_shape],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0], outs[1]),
        out_buffers=[new_sparse_indices_buf, new_shape_buf],
        name="sparse_reshape_cpu",
        tag="sparse_reshape_cpu",
    )
