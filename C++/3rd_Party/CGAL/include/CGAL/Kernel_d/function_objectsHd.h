// Copyright (c) 2000,2001  
// Utrecht University (The Netherlands),
// ETH Zurich (Switzerland),
// INRIA Sophia-Antipolis (France),
// Max-Planck-Institute Saarbruecken (Germany),
// and Tel-Aviv University (Israel).  All rights reserved. 
//
// This file is part of CGAL (www.cgal.org); you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 3 of the License,
// or (at your option) any later version.
//
// Licensees holding a valid commercial license may use this file in
// accordance with the commercial license agreement provided with the software.
//
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
// $URL$
// $Id: function_objectsHd.h,v 1.4 2017/02/06 12:37:06 Emmanuel.Pot Exp $
// 
//
// Author(s)     : Michael Seel

//---------------------------------------------------------------------
// file generated by notangle from noweb/function_objectsHd.lw
// please debug or modify noweb file
// coding: K. Mehlhorn, M. Seel
//---------------------------------------------------------------------

#ifndef CGAL_FUNCTION_OBJECTSHD_H
#define CGAL_FUNCTION_OBJECTSHD_H

#include <CGAL/basic.h>
#include <CGAL/enum.h>

namespace CGAL {

template <typename K>
class Compute_coordinateHd {
  typedef typename K::FT             FT;
  typedef typename K::Point_d        Point_d;
 public:
  typedef FT                         result_type;  
  result_type 
    operator()(const Point_d& p, int i) const
  {
    return p.cartesian(i);
  }
};

template <typename K>
class Point_dimensionHd {
  typedef typename K::RT             RT;
  typedef typename K::Point_d        Point_d;
  public:
  typedef int                       result_type;
  result_type 
    operator()(const Point_d& p) const
  {
    return p.dimension();
  }
};

template <typename K>
class Less_coordinateHd {
  typedef typename K::RT             RT;
  typedef typename K::Point_d        Point_d;
  public:
  typedef bool                       result_type;
  result_type 
    operator()(const Point_d& p, const Point_d& q, int i) const
  {
    int d = p.dimension();
    return p.cartesian(i)*q.homogeneous(d)<q.cartesian(i)*p.homogeneous(d);
  }
};


template <class R>
struct Lift_to_paraboloidHd {
typedef typename R::Point_d Point_d;
typedef typename R::RT RT;
typedef typename R::LA LA;

Point_d operator()(const Point_d& p) const
{ 
  int d = p.dimension();
  typename LA::Vector h(d+2);
  RT D = p.homogeneous(d);
  RT sum = 0;
  for (int i = 0; i<d; i++) {
    RT hi = p.homogeneous(i);
    h[i] = hi*D;
    sum += hi*hi;
  }
  h[d] = sum;
  h[d+1] = D*D;
  return Point_d(d+1,h.begin(),h.end());
}
};

template <class R>
struct Project_along_d_axisHd {
typedef typename R::Point_d Point_d;
typedef typename R::RT RT;
typedef typename R::LA LA;

Point_d operator()(const Point_d& p) const
{ int d = p.dimension();
  return Point_d(d-1, p.homogeneous_begin(),p.homogeneous_end()-2,
                 p.homogeneous(d));
}
};

template <class R>
struct MidpointHd {
typedef typename R::Point_d Point_d;
Point_d operator()(const Point_d& p, const Point_d& q) const
{ return Point_d(p + (q-p)/2); }
};

template <class R>
struct Center_of_sphereHd {
typedef typename R::Point_d Point_d;
typedef typename R::RT RT;
typedef typename R::LA LA;
template <class Forward_iterator>
Point_d operator()(Forward_iterator start, Forward_iterator end) const
{ CGAL_assertion(start!=end);
  CGAL_USE(end);
  int d = start->dimension();
  typename LA::Matrix M(d);
  typename LA::Vector b(d);
  Point_d pd = *start++;
  RT pdd  = pd.homogeneous(d);
  for (int i = 0; i < d; i++) { 
    // we set up the equation for p_i
    Point_d pi = *start++; 
    RT pid = pi.homogeneous(d);
    b[i] = 0;
    for (int j = 0; j < d; j++) {
      M(i,j) = RT(2) * pdd * pid * 
               (pi.homogeneous(j)*pdd - pd.homogeneous(j)*pid);
      b[i] += (pi.homogeneous(j)*pdd - pd.homogeneous(j)*pid) *
              (pi.homogeneous(j)*pdd + pd.homogeneous(j)*pid);
    }
  }
  RT D;
  typename LA::Vector x;
  LA::linear_solver(M,b,x,D);
  return Point_d(d,x.begin(),x.end(),D);
}

}; // Center_of_sphereHd


template <class R>
struct Squared_distanceHd {
typedef typename R::Point_d Point_d;
typedef typename R::Vector_d Vector_d;
typedef typename R::FT FT;
FT operator()(const Point_d& p, const Point_d& q) const
{ Vector_d v = p-q; return v.squared_length(); }
};

template <class R>
struct Position_on_lineHd {
typedef typename R::Point_d Point_d;
typedef typename R::LA LA;
typedef typename R::FT FT;
typedef typename R::RT RT;

bool operator()(const Point_d& p, const Point_d& s, const Point_d& t, 
     FT& l) const
{ int d = p.dimension(); 
  CGAL_assertion_msg((d==s.dimension())&&(d==t.dimension()&& d>0), 
  "position_along_line: argument dimensions disagree.");
  CGAL_assertion_msg((s!=t), 
  "Position_on_line_d: line defining points are equal.");
  RT lnum = (p.homogeneous(0)*s.homogeneous(d) - 
             s.homogeneous(0)*p.homogeneous(d)) * t.homogeneous(d); 
  RT lden = (t.homogeneous(0)*s.homogeneous(d) - 
             s.homogeneous(0)*t.homogeneous(d)) * p.homogeneous(d); 
  RT num(lnum), den(lden), lnum_i, lden_i; 
  for (int i = 1; i < d; i++) {  
    lnum_i = (p.homogeneous(i)*s.homogeneous(d) - 
              s.homogeneous(i)*p.homogeneous(d)) * t.homogeneous(d); 
    lden_i = (t.homogeneous(i)*s.homogeneous(d) - 
              s.homogeneous(i)*t.homogeneous(d)) * p.homogeneous(d); 
    if (lnum*lden_i != lnum_i*lden) return false; 
    if (lden_i != 0) { den = lden_i; num = lnum_i; }
  }
  l = R::make_FT(num,den);
  return true; 
}
};

template <class R>
struct Barycentric_coordinatesHd {
typedef typename R::Point_d Point_d;
typedef typename R::LA LA;
typedef typename R::RT RT;

template <class ForwardIterator, class OutputIterator>
OutputIterator operator()(ForwardIterator first, ForwardIterator last, 
  const Point_d& p, OutputIterator result)
{ TUPLE_DIM_CHECK(first,last,Barycentric_coordinates_d);
  CGAL_assertion_code( int d = p.dimension(); )
  typename R::Affine_rank_d affine_rank;
  CGAL_assertion(affine_rank(first,last)==d);
  typename LA::Matrix M(first,last);
  typename LA::Vector b(p.homogeneous_begin(),p.homogeneous_end()), x;
  RT D;
  LA::linear_solver(M,b,x,D);  
  for (int i=0; i< x.dimension(); ++result, ++i) {
    *result= R::make_FT(x[i],D); 
  }
  return result;
}
};


template <class R>
struct OrientationHd { 
typedef typename R::Point_d Point_d;
typedef typename R::LA LA;

template <class ForwardIterator>
Orientation operator()(ForwardIterator first, ForwardIterator last)
{ TUPLE_DIM_CHECK(first,last,Orientation_d);
  int d = static_cast<int>(std::distance(first,last)); 
  // range contains d points of dimension d-1
  CGAL_assertion_msg(first->dimension() == d-1,
  "Orientation_d: needs first->dimension() + 1 many points.");
  typename LA::Matrix M(d); // quadratic
  for (int i = 0; i < d; ++first,++i) {
    for (int j = 0; j < d; ++j) 
      M(i,j) = first->homogeneous(j); 
  }
  int row_correction = ( (d % 2 == 0) ? -1 : +1 );
  // we invert the sign if the row number is even i.e. d is odd
  return Orientation(row_correction * LA::sign_of_determinant(M));
}
};

template <class R>
struct Side_of_oriented_sphereHd { 
typedef typename R::Point_d Point_d;
typedef typename R::LA LA;
typedef typename R::RT RT;

template <class ForwardIterator> 
Oriented_side operator()(ForwardIterator first, ForwardIterator last, 
                         const Point_d& x)
{ 
  TUPLE_DIM_CHECK(first,last,Side_of_oriented_sphere_d);
  int d = static_cast<int>(std::distance(first,last)); // |A| contains |d| points
  CGAL_assertion_msg((d-1 == first->dimension()), 
  "Side_of_oriented_sphere_d: needs first->dimension()+1 many input points.");
  typename LA::Matrix M(d + 1); 
  for (int i = 0; i < d; ++first, ++i) { 
    RT Sum = 0;
    RT hd = first->homogeneous(d-1); 
    M(i,0) = hd*hd; 
    for (int j = 0; j < d; j++) { 
      RT hj = first->homogeneous(j); 
      M(i,j + 1) = hj * hd; 
      Sum += hj*hj; 
    }
    M(i,d) = Sum; 
  }
  RT Sum = 0; 
  RT hd = x.homogeneous(d-1); 
  M(d,0) = hd*hd; 
  for (int j = 0; j < d; j++) { 
    RT hj = x.homogeneous(j); 
    M(d,j + 1) = hj * hd; 
    Sum += hj*hj; 
  }
  M(d,d) = Sum; 
  return CGAL::Sign(- LA::sign_of_determinant(M));
}
};

template <class R>
struct Side_of_bounded_sphereHd { 
typedef typename R::Point_d Point_d;
typedef typename R::LA LA;
typedef typename R::RT RT;

template <class ForwardIterator> 
Bounded_side operator()(ForwardIterator first, ForwardIterator last, 
                        const Point_d& p)
{
  TUPLE_DIM_CHECK(first,last,region_of_sphere);
  typename R::Orientation_d _orientation;
  Orientation o = _orientation(first,last);
  CGAL_assertion_msg((o != 0), "Side_of_bounded_sphere_d: \
  A must be full dimensional.");
  typename R::Side_of_oriented_sphere_d _side_of_oriented_sphere;
  Oriented_side oside = _side_of_oriented_sphere(first,last,p);
  if (o == POSITIVE) {
    switch (oside) {
        case ON_POSITIVE_SIDE    :   return ON_BOUNDED_SIDE;
        case ON_ORIENTED_BOUNDARY:   return ON_BOUNDARY;
        case ON_NEGATIVE_SIDE    :   return ON_UNBOUNDED_SIDE;
    }       
  } else {
    switch (oside) {
        case ON_POSITIVE_SIDE    :   return ON_UNBOUNDED_SIDE;
        case ON_ORIENTED_BOUNDARY:   return ON_BOUNDARY;
        case ON_NEGATIVE_SIDE    :   return ON_BOUNDED_SIDE;
    }     
  }
  return ON_BOUNDARY; // never reached
}
};


template <class R>
struct Contained_in_simplexHd { 
typedef typename R::Point_d Point_d;
typedef typename R::LA LA;
typedef typename R::RT RT;

template <class ForwardIterator> 
bool operator()(ForwardIterator first, ForwardIterator last,
                const Point_d& p) 
{
  TUPLE_DIM_CHECK(first,last,Contained_in_simplex_d);
  int k = static_cast<int>(std::distance(first,last)); // |A| contains |k| points
  int d = first->dimension(); 
  CGAL_assertion_code(
    typename R::Affinely_independent_d check_independence; )
  CGAL_assertion_msg(check_independence(first,last),
    "Contained_in_simplex_d: A not affinely independent.");
  CGAL_assertion(d==p.dimension());

  typename LA::Matrix M(d + 1,k); 
  typename LA::Vector b(p.homogeneous_begin(),p.homogeneous_end()); 
  for (int j = 0; j < k; ++first, ++j) {
    for (int i = 0; i <= d; ++i)  
      M(i,j) = first->homogeneous(i); 
  }

  RT D; 
  typename LA::Vector lambda; 
  if ( LA::linear_solver(M,b,lambda,D) ) { 
    int s = CGAL_NTS sign(D); 
    for (int j = 0; j < k; j++) { 
      int t = CGAL_NTS sign(lambda[j]); 
      if (s * t < 0) return false; 
    }
    return true;
  }
  return false; 
}
};

template <class R>
struct Contained_in_affine_hullHd { 
typedef typename R::Point_d Point_d;
typedef typename R::LA LA;
typedef typename R::RT RT;

template <class ForwardIterator> 
bool operator()(ForwardIterator first, ForwardIterator last,
                const Point_d& p) 
{
  TUPLE_DIM_CHECK(first,last,Contained_in_affine_hull_d);
  int k = static_cast<int>(std::distance(first,last)); // |A| contains |k| points
  int d = first->dimension(); 
  typename LA::Matrix M(d + 1,k); 
  typename LA::Vector b(p.homogeneous_begin(),p.homogeneous_end()); 
  for (int j = 0; j < k; ++first, ++j) 
    for (int i = 0; i <= d; ++i) 
      M(i,j) = first->homogeneous(i); 
  return LA::is_solvable(M,b); 
}
};


template <class R>
struct Affine_rankHd { 
typedef typename R::Point_d Point_d;
typedef typename R::Vector_d Vector_d;
typedef typename R::LA LA;
typedef typename R::RT RT;

template <class ForwardIterator> 
int operator()(ForwardIterator first, ForwardIterator last) 
{
  TUPLE_DIM_CHECK(first,last,Affine_rank_d);
  int k = static_cast<int>(std::distance(first,last)); // |A| contains |k| points
  if (k == 0) return -1; 
  if (k == 1) return 0; 
  int d = first->dimension(); 
  typename LA::Matrix M(d,--k);
  Point_d p0 = *first; ++first; // first points to second
  for (int j = 0; j < k; ++first, ++j) {
    Vector_d v = *first - p0;
    for (int i = 0; i < d; i++) 
      M(i,j) = v.homogeneous(i); 
  }
  return LA::rank(M); 
}
};

template <class R>
struct Affinely_independentHd { 
typedef typename R::Point_d Point_d;
typedef typename R::LA LA;
typedef typename R::RT RT;

template <class ForwardIterator> 
bool operator()(ForwardIterator first, ForwardIterator last) 
{ typename R::Affine_rank_d rank; 
  int n = static_cast<int>(std::distance(first,last));
  return rank(first,last) == n-1;
}
};


template <class R>
struct Compare_lexicographicallyHd {
typedef typename R::Point_d Point_d;
typedef typename R::Point_d PointD; //MSVC hack
Comparison_result operator()(const Point_d& p1, const Point_d& p2)
{ return PointD::cmp(p1,p2); }
};

template <class R>
struct Contained_in_linear_hullHd {
typedef typename R::LA LA;
typedef typename R::RT RT;
typedef typename R::Vector_d Vector_d;

template<class ForwardIterator>
bool operator()(
  ForwardIterator first, ForwardIterator last, const Vector_d& x) 
{ TUPLE_DIM_CHECK(first,last,Contained_in_linear_hull_d);
  int k = static_cast<int>(std::distance(first,last)); // |A| contains |k| vectors
  int d = first->dimension(); 
  typename LA::Matrix M(d,k); 
  typename LA::Vector b(d); 
  for (int i = 0; i < d; i++) { 
     b[i] = x.homogeneous(i); 
     for (int j = 0; j < k; j++) 
       M(i,j) = (first+j)->homogeneous(i); 
  }
  return LA::is_solvable(M,b); 
}
};

template <class R>
struct Linear_rankHd {
typedef typename R::LA LA;
typedef typename R::RT RT;
template <class ForwardIterator>
int operator()(ForwardIterator first, ForwardIterator last)
{ TUPLE_DIM_CHECK(first,last,linear_rank);
  int k = static_cast<int>(std::distance(first,last)); // k vectors
  int d = first->dimension(); 
  typename LA::Matrix M(d,k); 
  for (int i = 0; i < d  ; i++)
     for (int j = 0; j < k; j++)  
       M(i,j) = (first + j)->homogeneous(i); 
  return LA::rank(M); 
}
};

template <class R>
struct Linearly_independentHd {
typedef typename R::LA LA;
typedef typename R::RT RT;
template <class ForwardIterator>
bool operator()(ForwardIterator first, ForwardIterator last)
{ typename R::Linear_rank_d rank;
  return rank(first,last) == static_cast<int>(std::distance(first,last));
}
};


template <class R>
struct Linear_baseHd {
typedef typename R::LA LA;
typedef typename R::RT RT;
typedef typename R::Vector_d Vector_d;
template <class ForwardIterator, class OutputIterator>
OutputIterator operator()(ForwardIterator first, ForwardIterator last,
  OutputIterator result)
{ TUPLE_DIM_CHECK(first,last,linear_base);
  int k = static_cast<int>(std::distance(first,last)); // k vectors
  int d = first->dimension(); 
  typename LA::Matrix M(d,k); 
  for (int j = 0; j < k; j++) 
    for (int i = 0; i < d; i++)
      M(i,j) = (first+j)->homogeneous(i); 

  std::vector<int> indcols; 
  int r = LA::independent_columns(M,indcols); 

  for (int l=0; l < r; l++) {
    typename LA::Vector v = M.column(indcols[l]);
    *result++ = Vector_d(d,v.begin(),v.end(),1);
  }
  return result; 
}
};

} //namespace CGAL

#endif // CGAL_FUNCTION_OBJECTSHD_H
