// Suite v1 (suite_v1.h): согласованность констант и golden. Диапазоны verify
// пересчитываются эталонным CPU-исполнителем этой сборки и сверяются с
// вшитыми golden — так MSVC/переносимая арифметика/новый компилятор, дающие
// другой результат, падают на ctest, а не на пользователе.
#include "catch.hpp"
#include "kernels.h"
#include "pi_digits.h"
#include "suite_v1.h"

#include <cstdint>

using namespace monkey;
using namespace monkey::suite_v1;

TEST_CASE("suite constants are complete and consistent", "[suite]") {
    REQUIRE(kTestCount == 4);
    for (const TestDef& t : kTests) {
        INFO("test " << t.name);
        for (int c = 0; c < kCategories; ++c) {
            const auto cat = static_cast<Category>(c);
            REQUIRE(t.size[c] > 0);
            REQUIRE(t.t_ref[c] >= 0); // 0 — suite ещё не откалиброван: баллов нет
            REQUIRE(t.golden[c].hits > 0);
            const KernelParams kp = make_params(t, cat);
            std::string why;
            REQUIRE(kernel_selfcheck(kp, &why));
            if (t.kernel == KernelId::Bbp) {
                REQUIRE(kp.bbp.block_terms == kBbpBlockTerms);
                REQUIRE(kp.bbp.pos0 + kp.bbp.positions + 8 <= pi_hex_count());
                REQUIRE(bbp_units(kp.bbp) < (1ull << 32)); // предел индекса в шейдере
            }
        }
    }
    const KernelParams fox = make_params(kTests[0], kCpuSt);
    REQUIRE(fox.n == 28);
    REQUIRE(fox.len() == 43);
}

TEST_CASE("verify ranges reproduce the embedded golden values", "[suite][golden]") {
    for (const VerifyCase& vc : kVerify) {
        const TestDef& t = kTests[vc.test];
        INFO("verify " << t.name << " [" << vc.begin << ", +" << vc.count << ")");
        const KernelParams kp = make_params(t, vc.category);
        const RangeResult r = cpu_run_range(kp, vc.begin, vc.count, nullptr);
        REQUIRE(r.count == vc.count);
        REQUIRE(r.hits == vc.golden.hits);
        REQUIRE(r.checksum == vc.golden.checksum);
    }
}
