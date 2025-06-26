#pragma once

#include <GeographicLib/TransverseMercator.hpp>
#include <geometry_msgs/msg/point.hpp>

class TransverseMercatorProjector {
public:
    // Structure to hold the origin (latitude and longitude)
    struct Origin {
        double lat; // Latitude of the origin
        double lon; // Longitude of the origin
    };

    // Constructor to initialize the projector with the origin
    explicit TransverseMercatorProjector(const Origin& origin)
        : origin_(origin), tm_(GeographicLib::Constants::WGS84_a(), 
                             GeographicLib::Constants::WGS84_f(), 0.9996) {}

    // Method to convert GPS coordinates (latitude, longitude, elevation) to a local ENU (East-North-Up) point
    geometry_msgs::msg::Point forward(double lat, double lon, double ele) const {
        double x, y; // Variables to store the projected coordinates
        // Perform the Transverse Mercator projection
        tm_.Forward(origin_.lon, lat, lon, x, y);

        // Create a Point message to store the projected coordinates
        geometry_msgs::msg::Point point;
        point.x = x; // Easting
        point.y = y; // Northing
        point.z = ele; // Elevation (unchanged)

        return point;
    }

    // Method to convert local ENU coordinates back to GPS coordinates (latitude, longitude)
    Origin reverse(double x, double y, double z) const {
        double lat, lon; // Variables to store the converted latitude and longitude
        // Perform the inverse Transverse Mercator projection
        tm_.Reverse(origin_.lon, x, y, lat, lon);

        // Return the converted latitude and longitude
        return {lat, lon};
    }

private:
    GeographicLib::TransverseMercator tm_; // Transverse Mercator projection object
    Origin origin_; // Origin (latitude and longitude) for the projection
};