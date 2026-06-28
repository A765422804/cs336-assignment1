FLOPs = 3516769894400 # batch_size = 1 的一次forward的浮点数运算次数
steps = 400000 # 迭代次数
batch_size = 1024
backward_FLOPs = 2 * FLOPs

total_FLOPs = steps * batch_size * (FLOPs + backward_FLOPs)

peak_FLOPs_per_second = 495e12 # FLOPs/s

MFU = 0.5

process_seconds = total_FLOPs / (MFU * peak_FLOPs_per_second)

print('process_hours: ', process_seconds / 3600)