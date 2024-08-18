import tensorflow as tf

tensor_zero_d =tf.constant(4)
print(tensor_zero_d)

tensor_bool = tf.constant([True,True,False])
print(tensor_bool)

tensor_string=tf.constant(["Hello","World"])
print(tensor_string)

tensor_one_d =tf.constant([2,0,-3], dtype=tf.float32)
casted_tensor_one_d = tf.cast(tensor_one_d,dtype=tf.int16)
print(tensor_one_d)
print(casted_tensor_one_d)

tensor_two_d =tf.constant([
    [1,2,0],
    [3,5, -1],
    [1,5,6],
    [2,3,8]
])
print(tensor_two_d)

tensor_three_d =tf.constant([
    [[1,2,0],
    [3,5, -1]],

    [[10,2,0],
    [1,0,2]],

    [[5,8,0],
    [2,7, 0]],

    [[2,1,9],
    [4,-3,32]]
])
print(tensor_three_d)

tensor_four_d =tf.constant([
[
    [
        [8, -3, 12],
        [5, 0, 7]
    ],
    [
        [14, 2, -6],
        [0, 9, -1]
    ],
    [
        [4, 5, -10],
        [3, -8, 11]
    ],
    [
        [7, -2, 18],
        [6, 3, 1]
    ]
],

[
    [
        [-4, 9, 5],
        [1, -7, 11]
    ],
    [
        [6, -3, 14],
        [-9, 0, 8]
    ],
    [
        [2, -6, 12],
        [4, -1, 7]
    ],
    [
        [3, 10, -5],
        [15, -12, 2]
    ]
],

[
    [
        [13, 7, -4],
        [-8, 2, 16]
    ],
    [
        [0, -5, 9],
        [3, 11, -6]
    ],
    [
        [-2, 14, 1],
        [7, -9, 4]
    ],
    [
        [6, -11, 0],
        [12, -7, 5]
    ]
]

])
print(tensor_four_d)

import numpy as np

np_array = np.array([1,2,4])
print(np_array)

converted_tensor = tf.convert_to_tensor(np_array)
print(converted_tensor)

eye_tensor = tf.eye(
    num_rows=3,
    num_columns=None,
    batch_shape=None,
    dtype=tf.dtypes.float32,
    name=None
)
print(eye_tensor)

fill_tensor = tf.fill(
    [3,4],5, name=None
)
print(fill_tensor)

zeros_tensor = tf.zeros(
    [3,2],
    dtype=tf.dtypes.float32,
    name=None,
    layout=None
)
print(zeros_tensor)

one_tensor = tf.ones(
    [5,3],
    dtype=tf.dtypes.float32,
    name=None,
    layout=None
)
print(one_tensor)

one_like_tensor = tf.ones_like(fill_tensor)
print(one_like_tensor)