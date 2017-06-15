// Copyright (c) 2009  INRIA Sophia-Antipolis (France).
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
// $Id: Classification_type.h,v 1.3 2016/09/23 08:36:05 Emmanuel.Pot Exp $
// 
//
// Author(s)     : Sebastien Loriot
//


#ifndef CGAL_INTERNAL_CLASSIFICATION_TYPE_H
#define CGAL_INTERNAL_CLASSIFICATION_TYPE_H

namespace CGAL{
  namespace internal{
    enum Classification_type {EXTERIOR,SINGULAR,REGULAR,INTERIOR};
  }
} 
#endif //CGAL_INTERNAL_CLASSIFICATION_TYPE_H
