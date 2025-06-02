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
    tm_.Forward(origin_.position.lon, gps.lat, gps.lon, x, y);
    return {x, y, gps.ele};
  }

  GPSPoint reverse(const BasicPoint3d& enu) const override {
    double lat, lon;
    tm_.Reverse(origin_.position.lon, enu.x(), enu.y(), lat, lon);
    return {lat, lon, enu.z()};
  }

 private:
  GeographicLib::TransverseMercator tm_;
  lanelet::Origin origin_;
};

}  // namespace projection
}  // namespace lanelet