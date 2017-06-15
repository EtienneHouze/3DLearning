// Copyright (c) 2013  The University of Western Sydney, Australia.
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
// $Id: Plane_scan_tree.h,v 1.2 2017/02/06 12:37:04 Emmanuel.Pot Exp $
//
//
// Author(s)     : Quincy Tse, Weisheng Si

/* Plane_scan_tree.h
 *
 * This header defines the class Plane_scan_tree, the data structure for the balanced search tree used in
 * the Narasimhan and Smid's algorithm for constructing Theta graph.
 * Implementations of members in this class can be found in Plane_scan_tree_impl.h.
 */

#ifndef CGAL_PLANE_SCAN_TREE_H
#define CGAL_PLANE_SCAN_TREE_H

#include <iostream>
#include <algorithm>
#include <functional>
#include <cstring>

#include <CGAL/Cone_spanners_2/Plane_scan_tree_impl.h>

namespace CGAL {

/* This namespace contains the internal implementation of the tree for builiding Theta graph.
 * This is not meant to be imported by other code.
 */
namespace ThetaDetail {

/*
 * This is the balanced tree structure needed in Narasimhan and Smid's book,
 * but implemented using a partial ternary B+ tree instead of red-black tree.
 *
 * This tree supports insert and search only, and, similar to a treap, is able
 * to determine not only the ordering according to a primary order specified
 * in the key_compare, but is also able to determine minimum weighted value
 * according to value_compare. Any single operation on this tree is guaranteed
 * to be at worst O(log n), and builds the tree from a list at O(nlogn).
 *
 * !!! Note: No two keys in the tree may have equal ordering
 * according to key_compare. Results are undefined if key_compare does not give
 * unique ordering of the inserted keys. The `Less_by_direction_2` functor implemented
 * by us guarantees the unique ordering, and is passed to Comp and VComp.
 *
 *  see  G. Narasimhan and M. Smid, Geometric Spanner Networks: Cambridge
 *        University Press, 2007, p. 71
 */
template <typename Key,
         typename T,
         typename Comp=std::less<Key>,
         typename VComp=std::less<const T>
         >
class Plane_scan_tree {
private:
    typedef _Node<Key, T, Comp, VComp>            _node_type;
    typedef typename  _node_type::_leaf_type      _leaf_type;
    typedef typename  _node_type::_internal_type  _internal_type;

public:
    typedef typename  _node_type::key_type      key_type;
    typedef typename  _node_type::mapped_type   mapped_type;
    typedef typename  _node_type::value_type    value_type;
    typedef typename  _node_type::key_compare   key_compare;
    typedef typename  _node_type::value_compare value_compare;

    typedef           value_type&               reference;
    typedef           const value_type&         const_reference;
    typedef           value_type*               pointer;
    typedef           const value_type*         const_pointer;

    typedef _Iterator<key_type, mapped_type, key_compare, value_compare>
    iterator;
    typedef _RIterator<key_type, mapped_type, key_compare, value_compare>
    reverse_iterator;
    typedef           iterator                  const_iterator;
    typedef           reverse_iterator          const_reverse_iterator;
    typedef           size_t                    size_type;

    /* Explicit Constructor. */
    explicit Plane_scan_tree (const key_compare& comp = key_compare(),
                              const value_compare& vcomp = value_compare())
        : less (comp), vless (vcomp), root (NULL), m_min (NULL),
          m_max (NULL), _size (0) {}

    /* Constructor */
    template <typename InputIterator>
    Plane_scan_tree (InputIterator first, InputIterator last,
                     const key_compare& comp = key_compare(),
                     const value_compare& vcomp = value_compare())
        : less (comp), vless (vcomp), root (NULL), m_min (NULL),
          m_max (NULL), _size (0)
    {
        // buids the tree
        /* Note: a more efficient algorithm building the tree bottom up may be
          worth implementing later */
        for (; first != last; ++first)
            add (first->first, first->second);
    }

    /* Destructor. This will recursively destroy all nodes in the tree, making
     * all iterators and pointers to values stored in this tree invalid.
     */
    ~Plane_scan_tree () {
        delete root;
        root = NULL;
        m_min = NULL;
        m_max = NULL;
        _size = 0;;
    }


    /* Returns the number of key-value pairs in the tree
     *
     * @return The number of key-value pairs in the tree.
     */
    size_t size() const {
        return _size;
    }

    /* Inserts a key-value pair into the tree.
     * @param k   The key
     * @param v   The value
     */
    void add (const key_type& k, const mapped_type& v) {
        if (NULL == root) {
            m_min = new _leaf_type (less, vless, this);
            m_max = m_min;
            root = m_min;
        }
        _leaf_type* l = root->leafNode(k);
        l->add(k, v);
        _size++;
    }

    /* find a key */
    iterator find(const key_type& k) {
        _leaf_type* l = root->leafNode(k);
        return iterator (l, k);
    }

    /* find a constant key */
    const_iterator find(const key_type& k) const {
        _leaf_type* l = root->leafNode(k);
        return const_iterator (l, k);
    }

    /* Returns the the minimum value that has a key strictly greater than
     * the specified key.
     *
     * @param x The threshold key
     * @return  The minimum value whose key is strictly greater than x.
     */
    const mapped_type* minAbove (const key_type& x) const {
        if (NULL == root) return NULL;
        return root->minAbove(x);
    }

    /* Begin Iterator */
    inline iterator begin() {
        return iterator (this->m_min);
    }

    /* Const Begin Iterator */
    inline const_iterator begin() const {
        return const_iterator (this->m_min);
    }

    /* End Iterator */
    inline iterator end() {
        static iterator res;

        return res;
    }

    /* Constant End Iterator */
    inline const_iterator end() const {
        static const_iterator res;

        return res;
    }

    /* Reverse order Begin Iterator */
    inline reverse_iterator rbegin() {
        return reverse_iterator (this->m_max);
    }

    /* Constant Reverse order Begin Iterator */
    const_reverse_iterator rbegin() const {
        return const_reverse_iterator (this->m_max);
    }

    /* Reverse order End Iterator */
    reverse_iterator rend() {
        static reverse_iterator res;

        return res;
    }

    /* Constant Reverse order End Iterator */
    const_reverse_iterator rend() const {
        static const_reverse_iterator res;

        return res;
    }

    friend class _Leaf<Key, T, Comp, VComp>;
    friend class _Internal<Key, T, Comp, VComp>;

    friend std::ostream& operator<< (std::ostream& os, const Plane_scan_tree<Key, T, Comp, VComp>& pst) {
        os << *pst.root << std::endl;

        return os;
    }

protected:
private:
    /* key_compare funtor */
    const key_compare less;

    /* value_compare funtor */
    const value_compare vless;

    /* pointer to root */
    _node_type* root;

    /* pointer to m_min */
    _leaf_type* m_min;

    /* pointer to m_max */
    _leaf_type* m_max;

    /* size of the tree */
    size_t _size;
};

}    // namespace ThetaDetail

}    // namespace CGAL

#endif
