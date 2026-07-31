#include "testing.h"
#include "../src/llama-context.h"

#include <iostream>

static void test_struct_exists(testing & t) {
    llama_gpipe_state state = {};
    t.assert_equal(0, state.n_stages);
    t.assert_equal(0, state.cur_stage);
    t.assert_equal(0, state.microbatch_size);
    t.assert_equal(false, state.enabled);
}

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("struct_exists", test_struct_exists);

    return t.summary();
}
