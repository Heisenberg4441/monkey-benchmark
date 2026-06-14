#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>

// Глобальные флаги и счетчики
std::atomic<bool> found(false);
std::atomic<unsigned long long> total_attempts(0);

// Вспомогательная функция для разбиения UTF-8 строки на символы
std::vector<std::string> split_utf8(const std::string& str) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < str.size()) {
        int char_len = 1;
        if ((str[i] & 0x80) == 0) char_len = 1;      // ASCII
        else if ((str[i] & 0xE0) == 0xC0) char_len = 2; // 2-byte UTF-8
        else if ((str[i] & 0xF0) == 0xE0) char_len = 3; // 3-byte UTF-8 (Кириллица)
        else if ((str[i] & 0xF8) == 0xF0) char_len = 4; // 4-byte UTF-8 (Эмодзи)
        chars.push_back(str.substr(i, char_len));
        i += char_len;
    }
    return chars;
}

// Режим 1: Случайная генерация (Infinite Monkey)
void random_worker(int thread_id, const std::vector<std::string>& target_chars, const std::vector<std::string>& alphabet) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, alphabet.size() - 1);

    size_t len = target_chars.size();
    // Буфер на стеке для максимальной скорости (без аллокаций в куче)
    std::vector<size_t> local_indices(len);

    while (!found.load(std::memory_order_relaxed)) {
        // Генерация
        for (size_t i = 0; i < len; ++i) {
            local_indices[i] = dist(gen);
        }
        
        // Проверка
        bool match = true;
        for (size_t i = 0; i < len; ++i) {
            if (alphabet[local_indices[i]] != target_chars[i]) {
                match = false;
                break;
            }
        }

        total_attempts.fetch_add(1, std::memory_order_relaxed);

        if (match) {
            found.store(true, std::memory_order_release);
            return;
        }
    }
}

// Режим 2: Brute Force (Последовательный перебор)
void brute_worker(int thread_id, int total_threads, const std::vector<std::string>& target_chars, const std::vector<std::string>& alphabet) {
    size_t N = alphabet.size();
    size_t len = target_chars.size();

    // Используем 128-битные числа для индекса, так как N^len очень быстро превышает лимит uint64_t
    // Для упрощения кода в C++ используем семантику переноса (поиск по диапазонам)
    unsigned long long chunk_size = 1000000; // Обрабатываем по 1 млн комбинаций за раз
    unsigned long long start_idx = thread_id * chunk_size;

    std::vector<size_t> current_combination(len, 0);

    while (!found.load(std::memory_order_relaxed)) {
        // Конвертируем число в N-ричную систему (комбинацию)
        unsigned long long temp_idx = start_idx;
        for (size_t i = 0; i < len; ++i) {
            current_combination[len - 1 - i] = temp_idx % N;
            temp_idx /= N;
        }

        for (unsigned long long i = 0; i < chunk_size; ++i) {
            if (found.load(std::memory_order_relaxed)) return;

            // Проверка
            bool match = true;
            for (size_t j = 0; j < len; ++j) {
                if (alphabet[current_combination[j]] != target_chars[j]) {
                    match = false;
                    break;
                }
            }

            total_attempts.fetch_add(1, std::memory_order_relaxed);

            if (match) {
                found.store(true, std::memory_order_release);
                return;
            }

            // Инкремент N-ричного числа
            for (size_t j = len; j-- > 0; ) {
                current_combination[j]++;
                if (current_combination[j] < N) break;
                current_combination[j] = 0;
            }
        }
        start_idx += total_threads * chunk_size;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Использование: " << argv[0] << " <reference.txt> <mode: random | brute>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::string mode = argv[2];

    // Чтение эталона
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть " << filename << "\n";
        return 1;
    }
    std::string reference((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (reference.empty()) {
        std::cerr << "Ошибка: файл пуст!\n";
        return 1;
    }

    // Формируем алфавит из уникальных символов эталона
    std::vector<std::string> target_chars = split_utf8(reference);
    std::vector<std::string> alphabet = target_chars;
    std::sort(alphabet.begin(), alphabet.end());
    alphabet.erase(std::unique(alphabet.begin(), alphabet.end()), alphabet.end());

    std::cout << "Эталон: " << reference << "\n";
    std::cout << "Длина (символов): " << target_chars.size() << "\n";
    std::cout << "Размер алфавита: " << alphabet.size() << "\n";
    std::cout << "Режим: " << mode << "\n";

    unsigned int num_threads = std::thread::hardware_concurrency();
    std::cout << "Потоков запущено: " << num_threads << "\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;

    if (mode == "random") {
        for (unsigned int i = 0; i < num_threads; ++i) {
            threads.emplace_back(random_worker, i, std::cref(target_chars), std::cref(alphabet));
        }
    } else if (mode == "brute") {
        for (unsigned int i = 0; i < num_threads; ++i) {
            threads.emplace_back(brute_worker, i, num_threads, std::cref(target_chars), std::cref(alphabet));
        }
    } else {
        std::cerr << "Неизвестный режим! Используйте 'random' или 'brute'\n";
        return 1;
    }

    // Вывод статистики в реальном времени
    while (!found.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - start_time).count();
        unsigned long long attempts = total_attempts.load();
        std::cout << "Прошло: " << elapsed << "с | Попыток: " << attempts << " | Скорость: " << (long long)(attempts / elapsed) << " оп/с\r" << std::flush;
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\n\n=== СОВПАДЕНИЕ НАЙДЕНО! ===\n";
    std::cout << "Затраченное время: " << total_time << " секунд\n";
    std::cout << "Всего попыток: " << total_attempts.load() << "\n";

    return 0;
}