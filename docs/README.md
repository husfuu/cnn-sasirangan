![image](https://github.com/husfuu/cnn-sasirangan/assets/70875733/92db1239-843e-416f-8a1b-277a7ebf57bd)


# Proses perhitungan output shape
Proses output shape dari setiap layer dalam model tersebut.
## Convolutional Layer 1:
* Input shape: (128, 128, 3) # Sesuai dengan parameter input_shape
* Output shape setelah konvolusi: (128 - 2 + 1, 128 - 2 + 1, 32) = (127, 127, 32)
* Output shape setelah MaxPooling: (127 // 2, 127 // 2, 32) = (63, 63, 32)

## Convolutional Layer 2:
* Input shape: (63, 63, 32) # Output dari layer sebelumnya
* Output shape setelah konvolusi: (63 - 3 + 1, 63 - 3 + 1, 64) = (61, 61, 64)
* Output shape setelah MaxPooling: (61 // 2, 61 // 2, 64) = (30, 30, 64)

## Flatten Layer:
* Input shape: (30, 30, 64) # Output dari layer sebelumnya
* Output shape setelah flatten: 30 * 30 * 64 = 57600

## Dense Layer (Output Layer):
* Input shape: 57600 # Output dari layer sebelumnya
* Output shape: 5 # Sesuai dengan parameter unit pada Dense layer.

# Perhitungan Total Parameters
- Parameter adalah bobot (weight) dan bias

## Convolutional Layer 1:
- ukuran kernel: (2, 2)
- jumlah saluran pada layer sebelumnya: 3 (RGB)
- parameter kernel = (ukuran kernel * jumlah saluran pada layer sebelumnya) + 1
- parameter kernel = (2 * 2 * 3) + 1 = 13
- total parameter untuk layer ini adalah 32 (jumlah filter) * 13 = 416

## Convolutional Layer 2:
- ukuran kernel: (2, 2)
- jumlah saluran pada layer sebelumnya: 32 (Convolutional Layer 1)
- parameter kernel = (ukuran kernel * jumlah saluran pada layer sebelumnya) + 1
- parameter kernel = (3 * 3 * 32) + 1 = 289
- total parameter untuk layer ini adalah 64 (jumlah filter) * 289 = 18496

## Dense Layer (Output Layer):
- input dari flaten layer: 57600
- jumlah unit pada Dense Layer (output): 5
- parameter bobot = input * unit output + bias
- parameter bobot = 57600 × 5 + 5 = 288000

