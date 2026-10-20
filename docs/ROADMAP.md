# ROADMAP: monkey-benchmark → бенчмарк suite v1

Рабочий план превращения «поиска с таймером» в сравнимый между машинами,
ОС и бэкендами бенчмарк. Постановка — `BENCHMARK_PROMPT.md`. Этот файл —
результат фазы 0 (аудит) и план фаз 1–5.

**Окружение аудита:** master `61bc279`, Apple M4 Pro (10P + 4E, 14 потоков,
24 ГБ), macOS 27.0, Apple clang 21.0, CMake 4.4, MoltenVK 1.4.2
(Vulkan 1.4.357, `shaderInt64 = true`, `shaderFloat64 = false`,
`shaderBufferInt64Atomics = false`). CUDA-устройства нет.

---

## 0. Аудит

### 0.1 Как воспроизводилось

- Чистая сборка `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release` (CPU + Vulkan),
  ctest 9/9 зелёный.
- Эталон для monkey — `The quick brown fox jumps over the lazy dog`
  (n=27, len=43), без перевода строки.
- Точечные пробы — одноразовые C++-программы поверх `src/` и
  `libmonkey_workloads.a` (в репозиторий не входят): маппинг кандидатов,
  checksum BBP, MR против решета Эратосфена, побитовый mulmod, BBP на d≈10⁶.
- Путь MSVC/CUDA (`struct Counter`, побитовый mulmod) эмулировался на clang
  сборкой с `-U__SIZEOF_INT128__`.
- Падение GPU-бэкенда имитировалось подменой `run_cuda`, которая печатает
  ошибку и делает `return` (ровно то, что делают `CUDA_OK`/`VK_OK`).

### 0.2 Известные дефекты — статус воспроизведения

| # | Дефект | Где | Воспроизведение → результат | Статус |
|---|---|---|---|---|
| 1 | GPU игнорирует `--workload` | `cuda_backend.cu:113`, `vulkan_backend.cpp:171` | `-gpu -w miller-rabin -d 3` → `"workload": "miller-rabin"`, `total_attempts = 17179869184` = 4 × 2³² — ровно столько же, сколько у `-gpu -w monkey`. CPU-MR за то же время: 172 M | ✅ Vulkan; CUDA — по коду |
| 2 | Разный маппинг counter → кандидат | `monkey_kernel.h:23`, `cuda_backend.cu:41,48`, `monkey.comp:31,36` | Проба: CPU counter=0 → `26 2 13 25 …`, GPU tid=0 k=0 → `26 8 11 16 …`. Теста на одинаковость кандидатов нет | ✅ |
| 2a | Пересечение потоков между диспатчами | `cuda_backend.cu:161`, `vulkan_backend.cpp:368` | Vulkan: `seed_d(t) = s + d·2¹⁹ + t·G mod 2³²`. На диспатче **8192** поток 0 получает seed потока диспатча 0 — полный повтор потока (~2.3 ч при диспатче ~1 с) | ✅ (расчёт) |
| 3 | Vulkan brute — 32-битный индекс | `monkey.comp:45`, `vulkan_backend.cpp:30,369` | `jihgfedcba` (код 9 876 543 210 > 2³²), `-gpu -m brute -d 10` → `solution_found: false`, «попыток» 90 G. Хуже, чем в описании: при batch 8192 шаг `pc.base += 524288·8192 = 2³² ≡ 0`, т.е. **каждый диспатч перебирает тот же [0, 2³²)** | ✅ |
| 4 | Vulkan завышает число попыток | `vulkan_backend.cpp:359` | `ba` (код 2): GPU → `total_attempts = 4294967296`, CPU → `3` | ✅ |
| 5 | `-all` дублирует работу | `cpu_backend.cpp:21`, `cuda_backend.cu:154`, `vulkan_backend.cpp:331` | Оба устройства стартуют с counter 0, общего диспетчера нет | ✅ по коду |
| 6 | `--duration` не соблюдается | `cpu_backend.cpp:24` | `-w bbp -d 2` жив через 40 с, убит. Батч 8192 цифр на поток × O(d log d) за цифру | ✅ |
| 7 | Стоимость операции растёт с counter | `workload_extra.cpp`, ядра | MR `-t 1`: 6.62 M/s (`-d 1`) → 6.04 (`-d 3`) → 5.75 (`-d 6`). Проба, нс на нечётное n (1 поток): 2²⁰ — 298, 2³² — 603, 2⁴⁸ — 838, 2⁶² — 713 | ✅ |
| 8 | Слишком длинные диспатчи Vulkan | `vulkan_backend.cpp:323` | `-gpu -d 10`: avg 5.22 G/s, **median 0**, CV 265 %; `-d 3`: CV 294 %. С `--batch-size 128`: 5.15 G/s, CV 4.0 % | ✅ |
| 9 | Прогрев — отдельный запуск с пересборкой; не те часы | `main.cpp:388–405`, `390, 409, 423, 455` | Прогрев создаёт новый `Control` и заново вызывает `run_cpu`/`run_vulkan` (instance, device, pipeline). В libstdc++ `high_resolution_clock` — это `system_clock` (не монотонные) | ✅ по коду |
| 10 | Зависание при сбое GPU | `main.cpp:421`, `cuda_backend.cu:20`, `vulkan_backend.cpp:34` | Имитация `CUDA_OK → return` с `-gpu` без `-d`: через 8 с процесс крутит `Попыток: 0`, убит вручную | ✅ (имитация) |
| 11 | Нет повторов и доверительных интервалов | `main.cpp:409–459` | Один прогон, сэмплы разгона/инициализации в статистике | ✅ по коду |
| 12 | Checksum BBP зависит от разбиения | `workload_extra.cpp:44`, `cpu_backend.cpp:36` | [0, 64) одним батчем → `119b1cf90c08f55e`, двумя по 32 → `000000041d93e9a4`. Нигде не печатается | ✅ |
| 13 | BBP ищет эталон относительно CWD; файл обязателен | `workload_extra.cpp:21`, `main.cpp:175` | Из другой директории: `verify() workload'а 'bbp' не прошёл`, exit 1. `monkey_bench -w bbp` → `Не указан reference_file` | ✅ |
| 14 | MR: `if (a >= n) continue` | `miller_rabin_kernel.h:82` | Полный перебор нечётных n < 1 795 265 022 против решета: текущее ядро ошибается на **2** числах — **4033** (37·109) и **4681** (31·151) объявлены простыми. С `a %= n; if (a == 0) continue` — 0 ошибок | ✅ реальный баг |
| 15 | MR: n < 2 не входит в iterations | `workload_extra.cpp:75` | `execute_batch(0, 10)` → `iterations = 8` | ✅ |
| 16 | MR: комментарий про свидетели не совпадает с кодом | `workload_extra.h:27–31` | Комментарий — {2…17} до 3.3·10¹⁴, код — всегда набор Синклера | ✅ |
| 17 | Побитовый mulmod (MSVC/CUDA) | `uint128.h:109–123`, `miller_rabin_kernel.h:21–39` | m ≥ 2⁶³: **96 992 из 100 000** результатов неверны (переполнение в `a <<= 1`). Скорость при m≈2⁶²: 5.5 нс (`__uint128_t`) против 64.7 нс — **×11.8** в изолированном микробенче (в постановке ×4.5 — вероятно, end-to-end; перемерю на MR после фикса) | ✅ |
| 18 | BBP: лишний 128-бит mulmod и FP64 | `bbp_kernel.h:24–68` | 8k+j ≤ 8d+6 < 2³² при d < 5.3·10⁸ → хватает 32-битного модуля. На M4 Pro `shaderFloat64 = false` — портировать double-ядро на Vulkan нельзя. Точность double-ядра у 10⁶: 16/16 цифр [999984, 10⁶) верны, **0.674 с на цифру** (1 поток), 0.054 с на d=10⁵ | ✅ |
| 19 | JSON неполный и ломается | `main.cpp:249–295` | Эталон с `\x01` → `json.decoder.JSONDecodeError: Invalid control character`. `-gpu` → `"threads": 14`. Нет CPU/GPU/ОС/компилятора/sha | ✅ |
| 20 | README расходится с кодом | `README.md:75, 123, 46/56/164–186` | `-march=native` «автоматически» (opt-in с 2d8714d); bbp/MR «being added»; локаль/UTF-8 якобы влияют на нагрузку (горячий путь работает с индексами, важны только n и len). Дополнительно: `:194–208` — «локальный буфер» и «carry-propagating increment» в brute (в коде деление на каждый кандидат), `:136` — «per-device breakdown» (есть только в строке прогресса) | ✅ |
| 21 | Пробелы в CI | `.github/workflows/*.yml` | Windows-тесты с `-DUSE_VULKAN=OFF`; нет Linux arm64; macOS-релиз CPU-only; нет кросс-бэкендового checksum-теста и валидации схемы | ✅ |

### 0.3 Новые дефекты (не из списка)

| # | Дефект | Где | Доказательство |
|---|---|---|---|
| N1 | **CUDA-бэкенд не компилируется.** `Counter gid = {{base_lo, base_hi}}` некорректно для `struct Counter {lo, hi}` | `cuda_backend.cu:75` | clang: `excess elements in scalar initializer`. Релизный прогон тега v0.4.0 (run 28226662962) упал на шаге Build ровно в CUDA-джобах (Linux, Windows); macOS без CUDA собрался. nvcc локально нет, поэтому причина «почти наверняка», а не доказана |
| N2 | **Переносимый `Counter` (MSVC/CUDA) неверно умножает.** `operator*(Counter, uint64_t)` берёт только младшие 32 бита множителя и ошибается при `a.hi ≠ 0` | `uint128.h:60–75` | Сборка с `-U__SIZEOF_INT128__`: `test_uint128` — 4 проваленных assert'а; brute на 128-битном пространстве не проходит `verify()`. На clang/gcc этот путь никогда не тестируется |
| N3 | **CI на master красный:** Windows, шаг Smoke bench, exit 139 (падение процесса) | `tests.yml` | run 28227748745. Логи истекли (HTTP 410), на macOS не воспроизводится. Причину ищу в фазе 4, на новом движке |
| N4 | Vulkan-бэкенд отбрасывает CPU-устройства (`score 0`) → на lavapipe в CI `vulkan_available() == false`, бенч молча уходит на CPU. Боевой Vulkan-путь в CI не исполняется ни разу | `vulkan_backend.cpp:74–81` | По коду; `test_vulkan_philox` берёт `devs[0]` без фильтра, поэтому он работает |
| N5 | brute на каноничной строке fox невозможен: 27⁴³ ≈ 2²⁰⁴ > 2¹²⁸ | `workload_monkey.cpp:17–25` | `fox.txt -m brute` → `verify() workload'а 'monkey' не прошёл` |
| N6 | Флаг `found` на GPU читается без синхронизации: в шейдере — неатомарное чтение в цикле, на хосте нет барьера SHADER_WRITE → HOST_READ; в CUDA — не-volatile `*found` | `monkey.comp:33,44`, `vulkan_backend.cpp:357–362`, `cuda_backend.cu:45,80` | По коду/спецификации Vulkan. Ранний выход не гарантирован |
| N7 | Явно запрошенный, но недоступный GPU (`--gpu-api cuda`) — тихий откат на CPU с `[warn]` | `main.cpp:357–370` | По коду. Нарушает критерий 4 («явная ошибка») |
| N8 | `MonkeyWorkload` вообще не пишет checksum — результат monkey не проверяется ничем | `workload_monkey.cpp:31–66` | По коду |
| N9 | Мелочи: `-t abc` молча = auto (`strtoul`); `pc.iters` обрезает `--batch-size` > 2³² до uint32; BBP `verify()` проверяет только первые 1024 цифры, а бенч считает любые d; файл π перечитывается при каждом `run_cpu` | `main.cpp:135`, `vulkan_backend.cpp:323`, `workload_extra.cpp:53` | По коду |

### 0.4 Базовые замеры master (для BENCH.md «до»)

M4 Pro, Release, без `-march=native`. Ops/sec — как печатает текущий бенч.

| Конфигурация | Результат |
|---|---|
| monkey random, fox, `-t 1 -d 5` | 131 M/s, CV 1.1 % |
| monkey random, fox, 14 потоков, `-d 5` | 1.28 G/s, CV 2.0 % |
| monkey random, fox, Vulkan, batch 8192, `-d 10` | avg 5.22 G/s, median 0, CV 265 % |
| monkey random, fox, Vulkan, batch 128, `-d 10` | 5.15 G/s, CV 4.0 % |
| monkey brute, fox | не запускается (N5) |
| monkey brute, `thequickbrownfox` (n=15, len=16), `-t 1` / 14 потоков | 294 M/s, CV 11.8 % / 3.40 G/s, CV 1.2 % |
| MR, `-t 1`, `-d 1 / 3 / 6` | 6.62 / 6.04 / 5.75 M/s |
| MR, 14 потоков, `-d 5` | 55.2 M/s, CV 4.9 % |
| BBP, 1 поток, double-ядро | 0.674 с/цифра при d≈10⁶ |

---

## 1. Ключевые проектные решения

### 1.1 Единый индекс работы и результат

- Каждый тест — это диапазон глобальных **64-битных** индексов `[0, N)` и
  чистая функция `result = f(params, i)`. Ядро header-only
  (`src/workloads/*_kernel.h`, `__host__ __device__`), рядом построчный
  GLSL-порт (`src/shaders/*.glsl`), как у `philox.h`/`philox.glsl`.
- Раздача по GPU-потокам: `i = begin + tid + step·total`, проверка `i < end`.
  Число потоков, размер диспатча и чанков на результат не влияет.
- **Checksum:** `Σ contrib(i, rᵢ) mod 2⁶⁴`, где
  `contrib(i, r) = r == 0 ? 0 : fmix64(fmix64(i) ^ r)` (финализатор splitmix64).
  Отклонение от формулы постановки: для «пустого» результата вклад нулевой.
  Причина — стоимость. У monkey и MR ≥ 95 % индексов дают r = 0 (нет
  совпадения префикса, составное число), а два `fmix64` (на GPU это 64-битное
  умножение на 32-битных половинах) стоили бы до 20–30 % времени самой
  операции. Пропуск единиц всё равно ловится: вместе с checksum сверяются
  `count` (сколько индексов обработано) и `hits` (сколько r ≠ 0).
- Сложение mod 2⁶⁴ коммутативно и ассоциативно, поэтому CPU-потоки, CUDA и
  Vulkan суммируют свои части, а хост складывает их. GPU копит uint64 в
  регистрах, редуцирует по workgroup в shared memory и пишет один слот на
  workgroup. 64-битные атомики не нужны: на MoltenVK их нет.

### 1.2 Workload'ы

**monkey-random.** Маппинг CPU становится каноничным:
`ctr = {pos, i_lo, i_hi, 0}`, `key = {seed, 0}`, символ = `v[0] mod n`
(Philox и `index_stream` не трогаю, KAT-тесты как есть). Результат —
длина совпавшего префикса `L ∈ [0, len]` с ранним выходом, в среднем ≈ 1.04
вызова Philox на кандидат. GPU-маппинг `seed + tid·G, {k, pos, 0, 0}`
удаляется.

**monkey-brute.** Кандидат — запись `i` в системе счисления n, дополненная
ведущими нулями до len. Результат — длина совпавшего суффикса (младший разряд
меняется быстрее всего), в среднем ≈ 1.04 деления на кандидат. Индекс
64-битный: 128-битный путь (`Counter`, N2) предлагаю **удалить**, см. вопрос 2.
В режиме custom «до совпадения»: если код эталона ≥ 2⁶⁴ — явная ошибка
(перебор 2⁶⁴ на 3.4 G/s — это 170 лет).

**bbp-pi — целочисленный fixed-point.**
- Дробь mod 1 хранится как uint64 (2⁻⁶⁴ ulp, естественное переполнение).
- Слагаемое `T = floor(2⁶⁴ · x / m)`. Здесь `x = 16^(d−k) mod m` считается
  32-битной Montgomery (64-битные произведения, без 128 бит), а деление —
  двумя шагами «столбиком» по 32-битным лимбам.
- Чётные знаменатели 8k+4 = 4(2k+1) и 8k+6 = 2(4k+3) сокращаются до
  нечётного модуля, дробь при этом точно та же.
- Ошибка ≤ 1 ulp на член, ≤ 8·10⁶ ≈ 2²³ ulp на позицию (веса 4+2+1+1). Верхние 32 бита (8 hex-цифр)
  надёжны, если не попасть ровно на границу — окно проверяется при сборке
  suite.
- **Гранулярность.** Одна позиция у 10⁶ — это ≈ 4·10⁶ членов, то есть
  0.2–0.7 с на ядро CPU и около 1 с на поток GPU. Такая единица ломает чанки
  по 10–20 мс и даёт риск TDR. Поэтому единица работы — **под-блок**
  `(позиция w, блок k из K = 4096 членов)`: ≈ 1 мс CPU. Под-блок возвращает
  частичную взвешенную сумму `4·S₁ − 2·S₄ − S₅ − S₆` mod 2⁶⁴. Суммы
  аккумулируются по позициям: на CPU — `fetch_add` на чанк, на CUDA —
  `atomicAdd(u64)`, на Vulkan — пара 32-битных `atomicAdd` с переносом.
  Хвост `k > d` (~16 членов) хост добавляет общей функцией ядра.
- **Проверка.** 8 hex-цифр каждой позиции сравниваются со встроенным в бинарь
  `pi_hex.txt`. Golden `Σ contrib(pos, digits)` считается из самого файла
  во время выполнения.
- Окно — последние позиции файла, `[999 992 − W, 999 992)`, чтобы хватило
  8 цифр.

**mr-prime64.** Нечётные `n = 2⁶² + 1 + 2i`. Каноничные свидетели Синклера с
`a %= n; if (a == 0) continue`. Одно ядро 64-битной Montgomery (REDC без
делений; R² mod n — удвоениями) с платформенным примитивом `mulhi64`:
- `__uint128_t` на GCC/Clang;
- `_umul128` / `__umulh` на MSVC;
- `__umul64hi` на CUDA;
- 4×`umulExtended` по 32-битным лимбам в GLSL.

Побитовый фоллбэк удаляется. Golden — число простых, `hits` и checksum. Окно
сверяется с сегментированным решетом: в тестах — окна 2²⁴…2⁴⁸ (быстро), окно
suite у 2⁶² — в тесте с меткой `slow` (базовые простые до 2³¹).

### 1.3 GLSL и 64 бита

- Ядра пишутся поверх тонкого слоя `u64.glsl`: `u64_add`, `u64_mul_lo`,
  `u64_mulhi`, `u64_divmod_u32`, … Слой собирается в двух вариантах SPIR-V:
  `U64_NATIVE` (`GL_EXT_shader_explicit_arithmetic_types_int64`, нужна
  `shaderInt64`) и `U64_EMU` (`uvec2`).
- Вариант выбирается в рантайме по фиче устройства и записывается в JSON.
- Тест паритета гоняет **оба** варианта на lavapipe и MoltenVK — оба
  поддерживают int64, поэтому эмуляция тоже проверяется на реальном
  устройстве.

### 1.4 Движок (фаза 2)

- **Сессия устройства** `IDevice`: `init()` (время пишется отдельно) →
  `run_range(test, begin, count) → {count, hits, checksum, seg_sums}` →
  `shutdown()`. Реализации: CPU (пул потоков живёт всю сессию), Vulkan,
  CUDA. Устройство создаётся один раз на весь прогон suite.
- **Диспетчер чанков** — одна `std::atomic<uint64_t> next`, `fetch_add` на
  чанк. Он общий для CPU-потоков и GPU в `-all`: дубликатов нет, балансировка
  получается сама.
- **Автотюнинг на прогреве:** чанк CPU ≈ 15 мс, диспатч GPU ≈ 30 мс (не
  больше 50 мс, далеко от TDR). На GPU одновременно в полёте 2 диспатча.
  Ожидание fence ограничено таймаутом (10 с) — по нему ошибка, а не вечное
  ожидание.
- **Fixed-work:** `t` — от старта первого чанка до завершения последнего,
  `steady_clock`. Повторы R (standard 3, quick 1) в той же сессии. Балл —
  медиана, в отчёт идут min, max и CV. CV > 3 % → `unstable`.
- **Timed/stress (`-d`):** сэмпл = завершённый чанк (метка времени, единицы).
  Stop срабатывает не позже чем через один чанк.
- **Ошибки:** исключение или код ошибки из любого устройства → `stop`, join,
  понятное сообщение, exit ≠ 0. Тест с «падающим» фейковым устройством.
- CPU-ST на macOS: поток с QoS `USER_INTERACTIVE`, чтобы планировщик держал
  его на P-ядре. На Linux/Windows привязки нет — описано в методологии.

### 1.5 Suite v1, баллы, версии

- Все константы — в `src/suite_v1.h`: тесты, параметры, размеры по
  категориям, `t_ref`, golden, диапазоны для `verify`.
- Размеры **разные по категориям** — каждый тест идёт 3–10 с на M4 Pro.
  Пример: monkey-random CPU-ST ≈ 0.8·10⁹, CPU-MT ≈ 8·10⁹, GPU ≈ 3·10¹⁰. Один
  размер не подходит: GPU-размер на CPU-ST шёл бы минуты.
- Размеры GPU одинаковы для всех API. `t_ref` GPU снимается на M4 Pro через
  Vulkan, CUDA-баллы считаются относительно него.
- Golden monkey и MR для каждого размера генерирует утилита
  `tools/suite_gen` (CPU-ядро; для MR дополнительно сегментированное решето).
  В бинарь она не входит. Golden BBP берётся из `pi_hex.txt`, который CMake
  встраивает в бинарь заголовком.
- Балл теста = `1000 · t_ref / t`, балл категории — геометрическое среднее
  тестов, итог — см. вопрос 1. Тест `invalid` (checksum/count/hits ≠ golden
  или цифра π не совпала) → баллов нет ни у теста, ни у его категории, ни
  итоговых.
- `suite_version = 1`. Любое изменение ядра, размера или golden поднимает
  версию.
- `custom`: Ops/sec без баллов, `comparable: false`.

### 1.6 CLI и вывод

- `monkey_bench` без аргументов = `run --standard`.
- Подкоманды:
  - `run [--quick|--standard|--stress] [--cpu|--gpu|--all] [--gpu-api …] [--json out.json]`;
  - `list`;
  - `info`;
  - `verify` — малые диапазоны каждого ядра на всех бэкендах и вариантах
    u64 против golden;
  - `custom …`.
- Старый синтаксис `<reference_file> -cpu -d 10` распознаётся как `custom`.
  Вывод на английском.
- JSON v2 пишет собственный небольшой writer: экранирование `< 0x20` как
  `\u00XX`, NaN/Inf → `null`. Вендорная зависимость не нужна. Поля — по
  постановке. Плюс `devices[].u64_mode`, `init_time_s`, `chunk_size` и флаги
  `debug_build`, `native_arch`, `custom`, `unstable`, `partial`.
- Системная информация: macOS — `sysctl`, `hw.perflevel*`; Linux —
  `/proc/cpuinfo`, `/sys/devices/system/cpu` (гибриды через
  `cpu_core`/`cpu_atom` и `cpu_capacity`); Windows — `__cpuid`,
  `GetLogicalProcessorInformationEx` (EfficiencyClass),
  `GlobalMemoryStatusEx`.
- Версия и git sha вшиваются на этапе configure.

### 1.7 CI и релизы (фаза 4)

- Тесты: ubuntu-latest x64 (lavapipe), ubuntu-24.04-arm (lavapipe),
  windows-latest (MSVC + Vulkan SDK), macos-latest arm64 (MoltenVK).
- Шаги на каждой ОС: ctest (с паритетом), `verify`,
  `run --quick --cpu --json` и `check-jsonschema`.
- GPU-размеры suite на lavapipe шли бы часами, поэтому в CI GPU-путь
  проверяется через `verify`.
- Отдельная задача — сборка с принудительно переносимой арифметикой
  (`MONKEY_PORTABLE_ARITH`): ловит класс багов N2 без MSVC.
- Windows без GPU: Vulkan-рантайм — lavapipe из mesa-dist-win. Не получится —
  Vulkan-тесты на Windows только собираются и честно помечаются как пропуск.
- Релизы:
  - Windows x64 и Linux x64 — CUDA + Vulkan;
  - Linux arm64 — Vulkan;
  - macOS arm64 — Vulkan, MoltenVK в архиве рядом с бинарём (`@rpath`).
    Не выйдет — проверенный откат на CPU с понятным сообщением.

  Везде без `-march=native`.

---

## 2. Фазы и коммиты

ctest зелёный после каждого коммита. Замеры «до/после» — в BENCH.md.

**Фаза 1 — паритет ядер и checksum**
1. `fix(mr): canonical witness reduction, count every n` — дефекты 14–16,
   тест против решета (n < 2²⁴ полностью, сегментированные окна).
2. `feat(kernels): shared arithmetic — mix64, mulhi64 per platform, Montgomery 32/64` + тесты, включая сборку `MONKEY_PORTABLE_ARITH`.
3. `perf(mr): Montgomery kernel, drop bit-serial fallback` — дефект 17, замер.
4. `feat(bbp): integer fixed-point kernel with k-block units` — дефекты 12, 13, 18. π встраивается в бинарь, проверка окна у 10⁶, замер.
5. `feat(monkey): canonical counter→candidate mapping and checksum` — дефекты 2, 3, N5, N8. Обновлённые тесты, в коммите — объяснение смены маппинга.
6. `feat(vulkan): GLSL ports of all kernels (int64 + uvec2), range executor` — дефекты 1, 4, N4, N6.
7. `feat(cuda): kernels from shared headers, range executor` — дефект N1. Только компиляция в CI, рантайм — на NVIDIA пользователя.
8. `test(parity): cross-backend checksum parity + partition independence`.

**Фаза 2 — движок**
1. `feat(engine): device sessions and atomic chunk dispatcher` (CPU).
2. `feat(engine): Vulkan/CUDA sessions — single init, 2 in flight, fence timeout`.
3. `feat(engine): fixed-work runner — warmup autotune, repeats, steady_clock` — дефекты 6–9, 11.
4. `fix(engine): backend errors stop the run and exit non-zero` — дефект 10 + тест.
5. `feat(engine): shared dispatcher for -all, timed mode on completed chunks` — дефект 5.
6. `bench: engine before/after`.

**Фаза 3 — suite v1, баллы, CLI, JSON v2**
1. `feat(suite): suite_v1.h, tools/suite_gen, embedded goldens`.
2. `feat(cli): run/list/info/verify/custom, legacy syntax → custom, English output` — дефект N7.
3. `feat(report): JSON v2 writer, system/build metadata, schema v2` — дефект 19.
4. `feat(suite): calibrate t_ref on M4 Pro, score table`.
5. `bench: run-to-run stability (≥ 5 прогонов standard, CV итога ≤ 3 %)`.

**Фаза 4 — CI и релизы** — дефекты 21, N3. Для проверки нужен пуш ветки
(не master) — спрошу разрешения перед первым пушем.

**Фаза 5 — документация** — README / README.ru (методология, как читать
баллы, ограничения), BENCH.md до/после, дефект 20.

---

## 3. Риски и честные ограничения

- **CUDA я не исполняю** — NVIDIA здесь нет. CUDA-код собирается в CI.
  Паритет на CUDA нужно один раз прогнать (`monkey_bench verify`) на машине
  с NVIDIA.
- **MSVC, Linux arm64 и Windows проверяются только в CI.** Поэтому в фазе 4
  нужны пуши ветки.
- **CV ≤ 3 %.**
  - Ноутбучный тротлинг на CPU-MT/GPU и P/E-планирование на CPU-ST — главный
    риск критерия 5.
  - Если медиана из 3 прогонов не уложится, варианты такие: длиннее тесты в
    пределах 3–10 с, пауза между тестами, порядок тестов. Решать буду по
    замерам.
- **Тесты GPU — фиксированного размера.** На слабом iGPU suite идёт заметно
  дольше 10 с на тест, как у y-cruncher/Cinebench. Прогресс печатается.
- **macOS-раннеры GitHub** — виртуалки с паравиртуальным GPU. MoltenVK там
  может не увидеть устройство, тогда Vulkan-часть на macOS CI — WARN.
- **BBP fixed-point:** окно suite подбирается так, чтобы ни одна позиция не
  попала на границу округления. Проверка — при генерации suite и в тесте.

---

## 4. Принятые решения (2026-10-20)

Ответ на план: «делай, не спрашивай подтверждение». Приняты рекомендованные
варианты:

1. **Итоговый балл** — `Overall = geomean(CPU-ST, CPU-MT, GPU)`, где GPU —
   лучший из доступных API. Баллы остальных API показываются отдельно. Без
   GPU Overall не считается: печатается `CPU = geomean(ST, MT)`, в JSON —
   флаг `partial`.
2. **128-битный brute удаляется.** Индекс везде 64-битный. В custom, если код
   эталона ≥ 2⁶⁴, — явная ошибка.
3. **Отклонения §1 приняты:**
   - нулевой вклад в checksum для r = 0;
   - BBP-единица — под-блок из 4096 членов, проверка 8 цифр на позицию;
   - размеры тестов по категориям;
   - GPU в CI — через `verify`.
4. **Пуш:** ветку для CI пушу сам, master — нет. Теги и релизы не создаю.
