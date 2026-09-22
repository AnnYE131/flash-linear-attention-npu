// CPU-only test: c++ -std=c++11 tests/cpp/test_chunk_state_contract.cpp -o /tmp/state_contract && /tmp/state_contract
#include "../../fla/ops/ascendc/common/chunk_state_contract.h"
#include <cassert>
#include <vector>

int main()
{
    int64_t chunks = -1;
    const fla::ChunkArrayView *none = nullptr;
    assert(fla::ValidateStateChunks(none, none, 2, 129, 64, chunks) && chunks == 3);
    assert(!fla::ValidateStateChunks(none, none, 1, 0, 64, chunks));
    assert(!fla::ValidateStateChunks(none, none, 1, 129, 0, chunks));
    std::vector<int64_t> cu{0, 1, 65, 130}, ci{0, 0, 1, 0, 2, 0, 2, 1};
    auto check = [&](bool withIndices = true, bool optional = false) {
        fla::ChunkArrayView a{cu.data(), cu.size()}, b{ci.data(), ci.size()};
        return fla::ValidateStateChunks(&a, withIndices ? &b : none, 1, 130, 64, chunks, optional);
    };
    assert(check() && chunks == 4); // ceil(total T/BT) would incorrectly give 3.
    assert(!check(false));
    assert(check(false, true) && chunks == 4);
    ci[0] = 2;
    assert(!check());
    ci[0] = 0;
    ci.push_back(2);
    assert(!check());
    ci.push_back(2);
    assert(!check());
    ci.resize(8);
    cu[1] = 0;
    assert(!check());
    cu[1] = 1;
    cu.back() = 129;
    assert(!check());
    cu.back() = 130;
    assert(check());
    cu = {0, 0, 65, 130};
    ci = {1, 0, 1, 1, 2, 0, 2, 1};
    fla::ChunkArrayView a{cu.data(), cu.size()}, b{ci.data(), ci.size()};
    assert(fla::ValidateStateChunks(&a, &b, 1, 130, 64, chunks, false, true) && chunks == 4);
    assert(!fla::ValidateStateChunks(&a, &b, 1, 130, 64, chunks));
}
