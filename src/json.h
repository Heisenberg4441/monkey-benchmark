#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Минимальный потоковый JSON-writer: корректное экранирование (включая
// управляющие символы < 0x20 как \u00XX), NaN/Inf -> null, запятые и отступы
// расставляются сами. 64-битные контрольные суммы пишутся строками "0x…":
// JSON-числа в JavaScript теряют точность выше 2^53.

namespace monkey {

class JsonWriter {
public:
    JsonWriter& begin_object() { return open('{'); }
    JsonWriter& end_object() { return close('}'); }
    JsonWriter& begin_array() { return open('['); }
    JsonWriter& end_array() { return close(']'); }

    // Ключ следующего значения внутри объекта.
    JsonWriter& key(const std::string& k) {
        sep();
        str(k);
        out_ += ": ";
        after_key_ = true;
        return *this;
    }

    JsonWriter& value(const std::string& v) { return raw_value([&] { str(v); }); }
    JsonWriter& value(const char* v) { return v ? value(std::string(v)) : null(); }
    JsonWriter& value(bool v) { return raw_value([&] { out_ += v ? "true" : "false"; }); }
    JsonWriter& value(double v) {
        if (!std::isfinite(v)) return null();
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.17g", v);
        return raw_value([&] { out_ += buf; });
    }
    JsonWriter& value(uint64_t v) { return raw_value([&] { out_ += std::to_string(v); }); }
    JsonWriter& value(int64_t v) { return raw_value([&] { out_ += std::to_string(v); }); }
    JsonWriter& value(unsigned v) { return value(static_cast<uint64_t>(v)); }
    JsonWriter& value(int v) { return value(static_cast<int64_t>(v)); }
    JsonWriter& null() { return raw_value([&] { out_ += "null"; }); }
    JsonWriter& hex(uint64_t v) {
        char buf[24];
        std::snprintf(buf, sizeof buf, "0x%016llx", static_cast<unsigned long long>(v));
        return value(std::string(buf));
    }

    // Сокращения для пар ключ-значение.
    template <class T>
    JsonWriter& kv(const std::string& k, const T& v) {
        key(k);
        return value(v);
    }
    // Пустая строка или ноль -> null.
    JsonWriter& kv_opt(const std::string& k, const std::string& v) {
        key(k);
        return v.empty() ? null() : value(v);
    }
    JsonWriter& kv_opt(const std::string& k, uint64_t v) {
        key(k);
        return v == 0 ? null() : value(v);
    }

    const std::string& str() const { return out_; }

private:
    template <class F>
    JsonWriter& raw_value(F write) {
        sep();
        write();
        return *this;
    }

    JsonWriter& open(char c) {
        sep();
        out_ += c;
        first_.push_back(true);
        return *this;
    }

    JsonWriter& close(char c) {
        const bool empty = first_.back();
        first_.pop_back();
        if (!empty) newline();
        out_ += c;
        if (first_.empty()) out_ += '\n';
        return *this;
    }

    // Разделитель перед элементом: запятая и перевод строки, кроме значения
    // сразу после ключа.
    void sep() {
        if (after_key_) {
            after_key_ = false;
            return;
        }
        if (first_.empty()) return;
        if (!first_.back()) out_ += ',';
        first_.back() = false;
        newline();
    }

    void newline() {
        out_ += '\n';
        out_.append(2 * first_.size(), ' ');
    }

    void str(const std::string& s) {
        out_ += '"';
        for (unsigned char c : s) {
            switch (c) {
            case '"': out_ += "\\\""; break;
            case '\\': out_ += "\\\\"; break;
            case '\b': out_ += "\\b"; break;
            case '\f': out_ += "\\f"; break;
            case '\n': out_ += "\\n"; break;
            case '\r': out_ += "\\r"; break;
            case '\t': out_ += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out_ += buf;
                } else {
                    out_ += static_cast<char>(c); // UTF-8 проходит как есть
                }
            }
        }
        out_ += '"';
    }

    std::string out_;
    std::vector<bool> first_;
    bool after_key_ = false;
};

} // namespace monkey
