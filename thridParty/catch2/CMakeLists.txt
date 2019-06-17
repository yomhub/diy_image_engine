# Copyright (C) 2018 Tommy Hinks <tommy.hinks@gmail.com>
# This file is subject to the license terms in the LICENSE file
# found in the top-level directory of this distribution.

set(tests
    ppm_io_test.cc
	pgm_io_test.cc
)

add_executable(thinks_pnm_io_test
    main.cc
    ${tests})
target_include_directories(thinks_pnm_io_test SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(thinks_pnm_io_test PUBLIC thinks_pnm_io)
set_target_properties(thinks_pnm_io_test PROPERTIES CXX_STANDARD 11)

add_test(NAME test COMMAND thinks_pnm_io_test)
