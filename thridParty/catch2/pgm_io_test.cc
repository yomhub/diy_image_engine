// Copyright(C) 2018 Tommy Hinks <tommy.hinks@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include <thinks/pnm_io/pnm_io.h>
#include <utils/catch_utils.h>

namespace pnm_io = thinks::pnm_io;

namespace {

void WriteInvalidPgmImage(std::ostream& os, std::string const& magic_number,
                          std::uint32_t const max_value,
                          std::size_t const width, std::size_t const height,
                          std::vector<std::uint8_t> const& pixel_data) {
  // Write header.
  os << magic_number << "\n"
     << width << "\n"
     << height << "\n"
     << max_value << "\n";  // Marks beginning of pixel data.

  // Write pixel data.
  os.write(reinterpret_cast<char const*>(pixel_data.data()), pixel_data.size());
}

}  // namespace

TEST_CASE("PGM - Write invalid filename throws") {
  auto const width = std::size_t{10};
  auto const height = std::size_t{10};
  auto const pixel_data = std::vector<std::uint8_t>(width * height * 3);
  auto const filename = std::string{};  // Invalid.

  // Not checking error message since it is OS dependent.
  REQUIRE_THROWS_AS(
      pnm_io::WritePgmImage(filename, width, height, pixel_data.data()),
      std::runtime_error);
}

TEST_CASE("PGM - Write invalid width throws") {
  auto const width = std::size_t{0};  // Invalid.
  auto const height = std::size_t{10};
  auto const pixel_data = std::vector<std::uint8_t>(width * height * 3);
  auto oss = std::ostringstream{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::WritePgmImage(oss, width, height, pixel_data.data()),
      std::invalid_argument,
      utils::ExceptionContentMatcher("width must be non-zero"));
}

TEST_CASE("PGM - Write invalid height throws") {
  auto const width = std::size_t{10};
  auto const height = std::size_t{0};  // Invalid.
  auto const pixel_data = std::vector<std::uint8_t>(width * height * 3);
  auto oss = std::ostringstream{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::WritePgmImage(oss, width, height, pixel_data.data()),
      std::invalid_argument,
      utils::ExceptionContentMatcher("height must be non-zero"));
}

TEST_CASE("PGM - Read invalid filename throws") {
  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  auto const filename = std::string{};  // Invalid.

  // Not checking error message since it is OS dependent.
  REQUIRE_THROWS_AS(
      pnm_io::ReadPgmImage(filename, &width, &height, &pixel_data),
      std::runtime_error);
}

TEST_CASE("PGM - Read invalid magic number throws") {
  auto ss = std::stringstream{};
  WriteInvalidPgmImage(ss,
                       "P4",  // Invalid.
                       255, 10, 10, std::vector<std::uint8_t>(10 * 10));

  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::ReadPgmImage(ss, &width, &height, &pixel_data),
      std::runtime_error,
      utils::ExceptionContentMatcher("magic number must be 'P5', was 'P4'"));
}

TEST_CASE("PGM - Read invalid width throws") {
  auto ss = std::stringstream{};
  WriteInvalidPgmImage(ss, "P5", 255,
                       0,  // Invalid.
                       10, std::vector<std::uint8_t>{});

  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::ReadPgmImage(ss, &width, &height, &pixel_data),
      std::runtime_error,
      utils::ExceptionContentMatcher("width must be non-zero"));
}

TEST_CASE("PGM - Read invalid height throws") {
  auto ss = std::stringstream{};
  WriteInvalidPgmImage(ss, "P5", 255, 10,
                       0,  // Invalid.
                       std::vector<std::uint8_t>{});

  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::ReadPgmImage(ss, &width, &height, &pixel_data),
      std::runtime_error,
      utils::ExceptionContentMatcher("height must be non-zero"));
}

TEST_CASE("PGM - Read invalid max value throws") {
  auto ss = std::stringstream{};
  WriteInvalidPgmImage(ss, "P5",
                       254,  // Invalid.
                       10, 10, std::vector<std::uint8_t>(10 * 10));

  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::ReadPgmImage(ss, &width, &height, &pixel_data),
      std::runtime_error,
      utils::ExceptionContentMatcher("max value must be 255, was 254"));
}

TEST_CASE("PGM - Read invalid file size throws") {
  auto ss = std::stringstream{};
  WriteInvalidPgmImage(ss, "P5", 255, 10, 10,
                       std::vector<std::uint8_t>(10 * 10 - 1));  // Invalid.

  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::ReadPgmImage(ss, &width, &height, &pixel_data),
      std::runtime_error,
      utils::ExceptionContentMatcher("failed reading 100 bytes"));
}

TEST_CASE("PGM - Round-trip") {
  auto const write_width = std::size_t{16};
  auto const write_height = std::size_t{16};
  auto write_pixels = std::vector<std::uint8_t>(write_width * write_height);
  auto pixel_index = std::size_t{0};
  for (auto i = std::size_t{0}; i < write_height; ++i) {
    for (auto j = std::size_t{0}; j < write_width; ++j) {
      write_pixels[pixel_index] = static_cast<std::uint8_t>(pixel_index);
      ++pixel_index;
    }
  }

  // Write image to IO stream.
  auto ss = std::stringstream{};
  pnm_io::WritePgmImage(ss, write_width, write_height, write_pixels.data());

  // Read image from IO stream.
  auto read_width = std::size_t{0};
  auto read_height = std::size_t{0};
  auto read_pixels = std::vector<std::uint8_t>{};
  pnm_io::ReadPgmImage(ss, &read_width, &read_height, &read_pixels);

  // Check that values were preserved.
  REQUIRE(read_width == write_width);
  REQUIRE(read_height == write_height);
  REQUIRE(read_pixels == write_pixels);
}
