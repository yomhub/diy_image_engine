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

void WriteInvalidPpmImage(std::ostream& os, std::string const& magic_number,
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

TEST_CASE("PPM - Write invalid filename throws") {
  auto const width = std::size_t{10};
  auto const height = std::size_t{10};
  auto const pixel_data = std::vector<std::uint8_t>(width * height * 3);
  auto const filename = std::string{};  // Invalid.

  // Not checking error message since it is OS dependent.
  REQUIRE_THROWS_AS(
      pnm_io::WritePpmImage(filename, width, height, pixel_data.data()),
      std::runtime_error);
}

TEST_CASE("PPM - Write invalid width throws") {
  auto const width = std::size_t{0};  // Invalid.
  auto const height = std::size_t{10};
  auto const pixel_data = std::vector<std::uint8_t>(width * height * 3);
  auto oss = std::ostringstream{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::WritePpmImage(oss, width, height, pixel_data.data()),
      std::invalid_argument,
      utils::ExceptionContentMatcher("width must be non-zero"));
}

TEST_CASE("PPM - Write invalid height throws") {
  auto const width = std::size_t{10};
  auto const height = std::size_t{0};  // Invalid.
  auto const pixel_data = std::vector<std::uint8_t>(width * height * 3);
  auto oss = std::ostringstream{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::WritePpmImage(oss, width, height, pixel_data.data()),
      std::invalid_argument,
      utils::ExceptionContentMatcher("height must be non-zero"));
}

TEST_CASE("PPM - Read invalid filename throws") {
  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  auto const filename = std::string{};  // Invalid.

  // Not checking error message since it is OS dependent.
  REQUIRE_THROWS_AS(
      pnm_io::ReadPpmImage(filename, &width, &height, &pixel_data),
      std::runtime_error);
}

TEST_CASE("PPM - Read invalid magic number throws") {
  auto ss = std::stringstream{};
  WriteInvalidPpmImage(ss,
                       "P5",  // Invalid.
                       255, 10, 10, std::vector<std::uint8_t>(10 * 10 * 3));

  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::ReadPpmImage(ss, &width, &height, &pixel_data),
      std::runtime_error,
      utils::ExceptionContentMatcher("magic number must be 'P6', was 'P5'"));
}

TEST_CASE("PPM - Read invalid width throws") {
  auto ss = std::stringstream{};
  WriteInvalidPpmImage(ss, "P6", 255,
                       0,  // Invalid.
                       10, std::vector<std::uint8_t>{});

  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::ReadPpmImage(ss, &width, &height, &pixel_data),
      std::runtime_error,
      utils::ExceptionContentMatcher("width must be non-zero"));
}

TEST_CASE("PPM - Read invalid height throws") {
  auto ss = std::stringstream{};
  WriteInvalidPpmImage(ss, "P6", 255, 10,
                       0,  // Invalid.
                       std::vector<std::uint8_t>{});

  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::ReadPpmImage(ss, &width, &height, &pixel_data),
      std::runtime_error,
      utils::ExceptionContentMatcher("height must be non-zero"));
}

TEST_CASE("PPM - Read invalid max value throws") {
  auto ss = std::stringstream{};
  WriteInvalidPpmImage(ss, "P6",
                       254,  // Invalid.
                       10, 10, std::vector<std::uint8_t>(10 * 10 * 3));

  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::ReadPpmImage(ss, &width, &height, &pixel_data),
      std::runtime_error,
      utils::ExceptionContentMatcher("max value must be 255, was 254"));
}

TEST_CASE("PPM - Read invalid file size throws") {
  auto ss = std::stringstream{};
  WriteInvalidPpmImage(ss, "P6", 255, 10, 10,
                       std::vector<std::uint8_t>(10 * 10 * 3 - 1));  // Invalid.

  auto width = std::size_t{0};
  auto height = std::size_t{0};
  auto pixel_data = std::vector<std::uint8_t>{};
  REQUIRE_THROWS_MATCHES(
      pnm_io::ReadPpmImage(ss, &width, &height, &pixel_data),
      std::runtime_error,
      utils::ExceptionContentMatcher("failed reading 300 bytes"));
}

TEST_CASE("PPM - Round-trip") {
  auto const write_width = std::size_t{64};
  auto const write_height = std::size_t{96};
  auto write_pixels = std::vector<std::uint8_t>(write_width * write_height * 3);
  auto pixel_index = std::size_t{0};
  for (auto i = std::size_t{0}; i < write_height; ++i) {
    for (auto j = std::size_t{0}; j < write_width; ++j) {
      write_pixels[pixel_index * 3 + 0] = static_cast<std::uint8_t>(i);
      write_pixels[pixel_index * 3 + 1] = static_cast<std::uint8_t>(j);
      write_pixels[pixel_index * 3 + 2] = static_cast<std::uint8_t>(i + j);
      ++pixel_index;
    }
  }

  // Write image to IO stream.
  auto ss = std::stringstream{};
  pnm_io::WritePpmImage(ss, write_width, write_height, write_pixels.data());

  // Read image from IO stream.
  auto read_width = std::size_t{0};
  auto read_height = std::size_t{0};
  auto read_pixels = std::vector<std::uint8_t>{};
  pnm_io::ReadPpmImage(ss, &read_width, &read_height, &read_pixels);

  // Check that values were preserved.
  REQUIRE(read_width == write_width);
  REQUIRE(read_height == write_height);
  REQUIRE(read_pixels == write_pixels);
}
