#pragma once
#include <lanelet2_io/Projection.h>
#include <GeographicLib/TransverseMercator.hpp>

namespace lanelet {
namespace projection {

class TransverseMercatorProjector : public Projector {
 public:
  explicit TransverseMercatorProjector(Origin origin)
      : Projector(origin), tm_(GeographicLib::Constants::WGS84_a(), 
      GeographicLib::Constants::WGS84_f(), 0.9996) {
    // Set the origin for the projection
    origin_ = origin;
  }

  BasicPoint3d forward(const GPSPoint& gps) const override {
    double x, y;
    // RCLCPP_INFO(rclcpp::get_logger("tm_debug"),
    //             "Input GPS: lat=%.8f, lon=%.8f, lon0=%.8f",
    //             gps.lat, gps.lon, origin_.position.lon);
    tm_.Forward(origin_.position.lon, gps.lat, gps.lon, x, y);
    // RCLCPP_INFO(rclcpp::get_logger("tm_debug"), "Projected X=%.3f, Y=%.3f", x, y);

    double y_origin, x_origin;
    tm_.Forward(origin_.position.lon, origin_.position.lat, origin_.position.lon, x_origin, y_origin);

    y -= y_origin;  // Shift Y so that origin latitude is at (0,0)

    // RCLCPP_INFO(rclcpp::get_logger("tm_debug"), "Corrected X=%.3f, Y=%.3f", x, y);

    // double lat_diff_m = (gps.lat - origin_.position.lat) * (GeographicLib::Constants::WGS84_a() * M_PI / 180.0);
    // RCLCPP_INFO(rclcpp::get_logger("tm_debug"), "Manual Y estimate: %.3f", lat_diff_m);

    return {x, y, gps.ele};
  }

  GPSPoint reverse(const BasicPoint3d& enu) const override {
    double lat, lon;
    tm_.Reverse(origin_.position.lon, enu.x(), enu.y(), lat, lon);
    return {lat, lon, enu.z()};
  }

  GeographicLib::TransverseMercator tm_;
  lanelet::Origin origin_;
};

}  // namespace projection
}  // namespace lanelet