// Copyright (c) 2008 GeometryFactory, Sophia Antipolis (France)
//  All rights reserved.
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
// $Id: gl.h,v 1.5 2017/02/06 12:37:01 Emmanuel.Pot Exp $

#ifndef CGAL_GL_H
#define CGAL_GL_H

#ifdef _MSC_VER
#  include <wtypes.h>
#  include <wingdi.h>
#endif

#ifdef __APPLE__
#  if TARGET_OS_IPHONE
#    include <OpenGLES/ES2/gl.h>
#  else
#    include <OpenGL/gl.h>
#  endif
#else
#  include <GL/gl.h>
#endif

#endif // CGAL_GL_H
