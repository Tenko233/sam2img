import random
for i in range(10):
    rgba = [random.randint(0, 255) for _ in range(3)]
    rgba.append(128)
    print(rgba)