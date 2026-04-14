# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

if("/home/sunyahui/ljk/FilterVector/FilterVectorCode/thirdparty/json-3.10.4.tar.gz" STREQUAL "")
  message(FATAL_ERROR "LOCAL can't be empty")
endif()

if(NOT EXISTS "/home/sunyahui/ljk/FilterVector/FilterVectorCode/thirdparty/json-3.10.4.tar.gz")
  message(FATAL_ERROR "File not found: /home/sunyahui/ljk/FilterVector/FilterVectorCode/thirdparty/json-3.10.4.tar.gz")
endif()

if("" STREQUAL "")
  message(WARNING "File cannot be verified since no URL_HASH specified")
  return()
endif()

if("" STREQUAL "")
  message(FATAL_ERROR "EXPECT_VALUE can't be empty")
endif()

message(VERBOSE "verifying file...
     file='/home/sunyahui/ljk/FilterVector/FilterVectorCode/thirdparty/json-3.10.4.tar.gz'")

file("" "/home/sunyahui/ljk/FilterVector/FilterVectorCode/thirdparty/json-3.10.4.tar.gz" actual_value)

if(NOT "${actual_value}" STREQUAL "")
  message(FATAL_ERROR "error:  hash of
  /home/sunyahui/ljk/FilterVector/FilterVectorCode/thirdparty/json-3.10.4.tar.gz
does not match expected value
  expected: ''
    actual: '${actual_value}'
")
endif()

message(VERBOSE "verifying file... done")
