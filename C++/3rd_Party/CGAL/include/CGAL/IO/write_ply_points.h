// Copyright (c) 2015  Geometry Factory
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
// You can redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// Licensees holding a valid commercial license may use this file in
// accordance with the commercial license agreement provided with the software.
//
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
// $URL$
// $Id: write_ply_points.h,v 1.4 2017/02/06 12:37:05 Emmanuel.Pot Exp $
//
// Author(s) : Simon Giraudot

#ifndef CGAL_WRITE_PLY_POINTS_H
#define CGAL_WRITE_PLY_POINTS_H

#include <CGAL/property_map.h>
#include <CGAL/point_set_processing_assertions.h>

#include <boost/version.hpp>

#include <iostream>
#include <iterator>

namespace CGAL {


//===================================================================================
/// \ingroup PkgPointSetProcessing
/// Saves the [first, beyond) range of points (positions + normals) to a .ply ASCII stream.
///
/// \pre normals must be unit vectors
///
/// @tparam ForwardIterator iterator over input points.
/// @tparam PointPMap is a model of `ReadablePropertyMap` with  value type `Point_3<Kernel>`.
///        It can be omitted if the value type of `ForwardIterator` is convertible to `Point_3<Kernel>`.
/// @tparam NormalPMap is a model of `ReadablePropertyMap` with a value type  `Vector_3<Kernel>`.
/// @tparam Kernel Geometric traits class.
///        It can be omitted and deduced automatically from the value type of `PointPMap`.
///
/// @return true on success.

// This variant requires all parameters.
template < typename ForwardIterator,
           typename PointPMap,
           typename NormalPMap,
           typename Kernel >
bool
write_ply_points_and_normals(
  std::ostream& stream, ///< output stream.
  ForwardIterator first,  ///< iterator over the first input point.
  ForwardIterator beyond, ///< past-the-end iterator over the input points.
  PointPMap point_pmap, ///< property map: value_type of ForwardIterator -> Point_3. 
  NormalPMap normal_pmap, ///< property map: value_type of ForwardIterator -> Vector_3. 
  const Kernel& /*kernel*/) ///< geometric traits.
{
  CGAL_point_set_processing_precondition(first != beyond);

  if(!stream)
  {
    std::cerr << "Error: cannot open file" << std::endl;
    return false;
  }

  // Write header
  stream << "ply" << std::endl
         << "format ascii 1.0" << std::endl
         << "comment Generated by the CGAL library" << std::endl
         << "element vertex " << std::distance (first, beyond) << std::endl
         << "property double x" << std::endl
         << "property double y" << std::endl
         << "property double z" << std::endl
         << "property double nx" << std::endl
         << "property double ny" << std::endl
         << "property double nz" << std::endl
         << "end_header" << std::endl;
  

  // Write positions + normals
  for(ForwardIterator it = first; it != beyond; it++)
  {
    stream << get(point_pmap, *it) << " "
           << get(normal_pmap, *it) << std::endl;
  }

  return ! stream.fail();
}

/// @cond SKIP_IN_MANUAL
// This variant deduces the kernel from the point property map.
template < typename ForwardIterator,
           typename PointPMap,
           typename NormalPMap >
bool
write_ply_points_and_normals(
  std::ostream& stream, ///< output stream.
  ForwardIterator first, ///< first input point.
  ForwardIterator beyond, ///< past-the-end input point.
  PointPMap point_pmap, ///< property map: value_type of OutputIterator -> Point_3.
  NormalPMap normal_pmap) ///< property map: value_type of OutputIterator -> Vector_3.
{
  typedef typename boost::property_traits<PointPMap>::value_type Point;
  typedef typename Kernel_traits<Point>::Kernel Kernel;
  return write_ply_points_and_normals(
    stream,
    first, beyond,
    point_pmap,
    normal_pmap,
    Kernel());
}
/// @endcond

/// @cond SKIP_IN_MANUAL
// This variant creates a default point property map = Identity_property_map.
template <typename ForwardIterator,
          typename NormalPMap
>
bool
write_ply_points_and_normals(
  std::ostream& stream, ///< output stream.
  ForwardIterator first, ///< first input point.
  ForwardIterator beyond, ///< past-the-end input point.
  NormalPMap normal_pmap) ///< property map: value_type of OutputIterator -> Vector_3.
{
  return write_ply_points_and_normals(
    stream,
    first, beyond,
    make_identity_property_map(
    typename std::iterator_traits<ForwardIterator>::value_type()),
    normal_pmap);
}
/// @endcond


//===================================================================================
/// \ingroup PkgPointSetProcessing
/// Saves the [first, beyond) range of points (positions only) to a .ply ASCII stream.
///
/// @tparam ForwardIterator iterator over input points.
/// @tparam PointPMap is a model of `ReadablePropertyMap` with a value_type = `Point_3<Kernel>`.
///        It can be omitted if the value type of `ForwardIterator` is convertible to `Point_3<Kernel>`.
/// @tparam Kernel Geometric traits class.
///        It can be omitted and deduced automatically from the value type of `PointPMap`.
///
/// @return true on success.

// This variant requires all parameters.
template < typename ForwardIterator,
           typename PointPMap,
           typename Kernel >
bool
write_ply_points(
  std::ostream& stream, ///< output stream.
  ForwardIterator first,  ///< iterator over the first input point.
  ForwardIterator beyond, ///< past-the-end iterator over the input points.
  PointPMap point_pmap, ///< property map: value_type of ForwardIterator -> Point_3.
  const Kernel& ) ///< geometric traits.
{
  // basic geometric types
  typedef typename Kernel::Point_3 Point;

  CGAL_point_set_processing_precondition(first != beyond);

  if(!stream)
  {
    std::cerr << "Error: cannot open file" << std::endl;
    return false;
  }

  // Write header
  stream << "ply" << std::endl
         << "format ascii 1.0" << std::endl
         << "comment Generated by the CGAL library" << std::endl
         << "element vertex " << std::distance (first, beyond) << std::endl
         << "property double x" << std::endl
         << "property double y" << std::endl
         << "property double z" << std::endl
         << "end_header" << std::endl;

  // Write positions
  for(ForwardIterator it = first; it != beyond; it++)
  {
    Point p = get(point_pmap, *it);
    stream << p << std::endl;
  }

  return ! stream.fail();
}

/// @cond SKIP_IN_MANUAL
// This variant deduces the kernel from the point property map.
template < typename ForwardIterator,
           typename PointPMap >
bool
write_ply_points(
  std::ostream& stream, ///< output stream.
  ForwardIterator first, ///< first input point.
  ForwardIterator beyond, ///< past-the-end input point.
  PointPMap point_pmap) ///< property map: value_type of OutputIterator -> Point_3.
{
  typedef typename boost::property_traits<PointPMap>::value_type Point;
  typedef typename Kernel_traits<Point>::Kernel Kernel;
  return write_ply_points(
    stream,
    first, beyond,
    point_pmap,
    Kernel());
}
/// @endcond

/// @cond SKIP_IN_MANUAL
// This variant creates a default point property map = Identity_property_map.
template < typename ForwardIterator >
bool
write_ply_points(
  std::ostream& stream, ///< output stream.
  ForwardIterator first, ///< first input point.
  ForwardIterator beyond) ///< past-the-end input point.
{
  return write_ply_points(
    stream,
    first, beyond,
    make_identity_property_map(
    typename std::iterator_traits<ForwardIterator>::value_type())
    );
}
/// @endcond


} //namespace CGAL

#endif // CGAL_WRITE_PLY_POINTS_H