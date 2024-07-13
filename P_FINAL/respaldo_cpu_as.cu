int getMax(const std::vector<int>& arr) {
    int max = arr[0];
    for (int num : arr) {
        if (num > max) {
            max = num;
        }
    }
    return max;
}

void countingSort(std::vector<int>& arr, int exp) {
    int n = arr.size();
    std::vector<int> output(n);
    int count[10] = {0};

    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

void radixSort(std::vector<int>& arr) {
    int max = getMax(arr);

    for (int exp = 1; max / exp > 0; exp *= 10) {
        countingSort(arr, exp);
    }
}

void filtro_mediana_cpu(float* img_in, float* img_out, int width, int height, int W) {
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        for (int pixel = 0; pixel < width * height; pixel++) {
            int x = pixel % width;
            int y = pixel / width;
            float window[(2 * W + 1) * (2 * W + 1)];
            int count = 0;
            for (int i = x - W; i <= x + W; i++) {
                for (int j = y - W; j <= y + W; j++) {
                    if (i >= 0 && i < width && j >= 0 && j < height) {
                        window[count++] = img_in[j * width + i];
                    }
                }
            }

            // Escalar a enteros
            std::vector<int> int_window(count);
            for (int k = 0; k < count; k++) {
                int_window[k] = static_cast<int>(window[k] * 1000); // Ajustar escala si es necesario
            }

            // Aplicar Radix Sort
            radixSort(int_window);

            // Desescalar a flotantes y encontrar la mediana
            for (int k = 0; k < count; k++) {
                window[k] = static_cast<float>(int_window[k]) / 1000.0f; // Ajustar escala si es necesario
            }

            img_out[pixel] = window[count / 2];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    printf("Tiempo CPU: %f\n", duration.count());
}