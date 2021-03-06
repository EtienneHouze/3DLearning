// Copyright (c) 2016 GeometryFactory
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
// $Id: Handle_IO_for_pair_of_int.h,v 1.1 2016/09/23 08:36:06 Emmanuel.Pot Exp $
//
//
// Author(s)     : Laurent Rineau

#ifndef CGAL_INTERNAL_MESH_3_INTERNAL_HANDLE_IO_FOR_PAIR_OF_INT_H
#define CGAL_INTERNAL_MESH_3_INTERNAL_HANDLE_IO_FOR_PAIR_OF_INT_H

#include <utility>
#include <CGAL/Mesh_3/io_signature.h>
#include <ostream>
#include <istream>

namespace CGAL {
template <>
struct Get_io_signature<std::pair<int, int> > {
  std::string operator()() const
  {
    return std::string("std::pair<i,i>");
  }
}; // end Get_io_signature<std::pair<int, int> >

inline std::ostream& operator<<(std::ostream& out, const std::pair<int, int>& id) {
  return out << id.first << " " << id.second;
}
inline std::istream& operator>>(std::istream& in, std::pair<int, int>& id) {
  return in >> id.first >> id.second;
}

template <>
class Output_rep<std::pair<int, int> > : public IO_rep_is_specialized {
  typedef std::pair<int, int> T;
  const T& t;
public:
  //! initialize with a const reference to \a t.
  Output_rep( const T& tt) : t(tt) {}
  //! perform the output, calls \c operator\<\< by default.
  std::ostream& operator()( std::ostream& out) const {
    if(is_ascii(out)) {
      out << t.first << " " << t.second;
    } else {
      CGAL::write(out, t.first);
      CGAL::write(out, t.second);
    }
    return out;
  }
};

template <>
class Input_rep<std::pair<int, int> > : public IO_rep_is_specialized {
  typedef std::pair<int, int> T;
  T& t;
public:
  //! initialize with a const reference to \a t.
  Input_rep( T& tt) : t(tt) {}
  //! perform the output, calls \c operator\<\< by default.
  std::istream& operator()( std::istream& in) const {
    if(is_ascii(in)) {
      in >> t.first >> t.second;
    } else {
      CGAL::read(in, t.first);
      CGAL::read(in, t.second);
    }
    return in;
  }
};
} // end namespace CGAL

#endif // CGAL_INTERNAL_MESH_3_INTERNAL_HANDLE_IO_FOR_PAIR_OF_INT_H
