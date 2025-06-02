#ifndef TWIST_WITH_COVARIANCE_STAMPED_HPP
#define TWIST_WITH_COVARIANCE_STAMPED_HPP

#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>

namespace tf2 {

template <>
inline void doTransform(
  const geometry_msgs::msg::TwistWithCovarianceStamped &in,
  geometry_msgs::msg::TwistWithCovarianceStamped &out,
  const geometry_msgs::msg::TransformStamped &transform
) {
  // Transform the header
  out.header.stamp = in.header.stamp;
  out.header.frame_id = transform.header.frame_id;

  // Transform the twist (linear and angular velocities)
  tf2::doTransform(in.twist.twist.linear, out.twist.twist.linear, transform);
  tf2::doTransform(in.twist.twist.angular, out.twist.twist.angular, transform);

  // Copy the covariance (no transformation needed)
  out.twist.covariance = in.twist.covariance;
}

} // namespace tf2

#endif // TWIST_WITH_COVARIANCE_STAMPED_HPP