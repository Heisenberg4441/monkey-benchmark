// JsonWriter: экранирование (в т.ч. управляющих символов, из-за которых
// старый ручной JSON был невалиден), NaN/Inf -> null, структура и запятые.
#include "catch.hpp"
#include "json.h"

#include <cmath>
#include <limits>
#include <string>

using monkey::JsonWriter;

TEST_CASE("JSON strings are escaped", "[json]") {
    JsonWriter w;
    w.begin_object().kv("s", std::string("a\"b\\c\n\t\x01\x1f end")).end_object();
    REQUIRE(w.str() == "{\n  \"s\": \"a\\\"b\\\\c\\n\\t\\u0001\\u001f end\"\n}\n");
}

TEST_CASE("UTF-8 passes through unchanged", "[json]") {
    JsonWriter w;
    w.begin_object().kv("ru", std::string("Привет 🐒")).end_object();
    REQUIRE(w.str().find("Привет 🐒") != std::string::npos);
}

TEST_CASE("non-finite numbers become null", "[json]") {
    JsonWriter w;
    w.begin_array()
        .value(std::numeric_limits<double>::quiet_NaN())
        .value(std::numeric_limits<double>::infinity())
        .value(1.5)
        .end_array();
    REQUIRE(w.str() == "[\n  null,\n  null,\n  1.5\n]\n");
}

TEST_CASE("nested structure, hex and optional values", "[json]") {
    JsonWriter w;
    w.begin_object();
    w.kv("n", uint64_t{18446744073709551615ull});
    w.key("h").hex(0xabcull);
    w.kv_opt("empty", std::string());
    w.key("arr").begin_array().end_array();
    w.key("obj").begin_object().kv("b", true).end_object();
    w.end_object();
    REQUIRE(w.str() ==
            "{\n  \"n\": 18446744073709551615,\n  \"h\": \"0x0000000000000abc\",\n  \"empty\": null,\n"
            "  \"arr\": [],\n  \"obj\": {\n    \"b\": true\n  }\n}\n");
}
