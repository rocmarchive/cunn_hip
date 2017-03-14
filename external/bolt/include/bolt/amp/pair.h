/***************************************************************************
*   � 2012,2014 Advanced Micro Devices, Inc. All rights reserved.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/



/*! \file bolt/amp/pair.h
 *  \brief A type encapsulating a heterogeneous pair of elements
 */

#pragma once
#if !defined( BOLT_AMP_PAIR_H )
#define BOLT_AMP_PAIR_H

#include <iterator>
#include <type_traits>
#include <numeric>

namespace bolt
{

    namespace amp
    {
/*! \addtogroup Miscellaneous
 *  \{
 */

/*! \addtogroup pair
 *  \{
 */

/*! \p pair is a generic data structure encapsulating a heterogeneous
 *  pair of values.
 *
 *  \tparam T1 The type of \p pair's first object type.  There are no
 *          requirements on the type of \p T1. <tt>T1</tt>'s type is
 *          provided by <tt>pair::first_type</tt>.
 *
 *  \tparam T2 The type of \p pair's second object type.  There are no
 *          requirements on the type of \p T2. <tt>T2</tt>'s type is
 *          provided by <tt>pair::second_type</tt>.
 */
template <typename T1, typename T2>
  struct pair
{
  /*! \p first_type is \p pair's first object type.
   */
  typedef T1 first_type;

  /*! \p second_type is \p pair's second object type.
   */
  typedef T2 second_type;

  /*! The \p pair's first object.
   */
  first_type first;

  /*! The \p pair's second object.
   */
  second_type second;

  /*! \p pair's default constructor constructs \p first
   *  and \p second using \c first_type & \c second_type's
   *  default constructors, respectively.
   */
  pair(void) restrict(cpu, amp);

  /*! This constructor accepts two objects to copy into this \p pair.
   *
   *  \param x The object to copy into \p first.
   *  \param y The object to copy into \p second.
   */
  pair(const T1 &x, const T2 &y) restrict(cpu, amp);

  /*! This copy constructor copies from a \p pair whose types are
   *  convertible to this \p pair's \c first_type and \c second_type,
   *  respectively.
   *
   *  \param p The \p pair to copy from.
   *
   *  \tparam U1 is convertible to \c first_type.
   *  \tparam U2 is convertible to \c second_type.
   */
  template <typename U1, typename U2>
  pair(const pair<U1,U2> &p) restrict(cpu, amp);

  /*! This copy constructor copies from a <tt>std::pair</tt> whose types are
   *  convertible to this \p pair's \c first_type and \c second_type,
   *  respectively.
   *
   *  \param p The <tt>std::pair</tt> to copy from.
   *
   *  \tparam U1 is convertible to \c first_type.
   *  \tparam U2 is convertible to \c second_type.
   */
  template <typename U1, typename U2>
  pair(const std::pair<U1,U2> &p) restrict(cpu, amp);

  ~pair() restrict(amp, cpu) {};

}; // end pair


/*! This operator tests two \p pairs for equality.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>x.first == y.first && x.second == y.second</tt>.
 *
 *  \tparam T1 is a model of <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">Equality Comparable</a>.
 *  \tparam T2 is a model of <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">Equality Comparable</a>.
 */
template <typename T1, typename T2>
    static
    inline
    bool operator==(const pair<T1,T2> &x, const pair<T1,T2> &y) restrict(cpu, amp);


/*! This operator tests two pairs for ascending ordering.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>x.first < y.first || (!(y.first < x.first) && x.second < y.second)</tt>.
 *
 *  \tparam T1 is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 *  \tparam T2 is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 */
template <typename T1, typename T2>
    static
    inline
    bool operator<(const pair<T1,T2> &x, const pair<T1,T2> &y) restrict(cpu, amp);


/*! This operator tests two pairs for inequality.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>!(x == y)</tt>.
 *
 *  \tparam T1 is a model of <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">Equality Comparable</a>.
 *  \tparam T2 is a model of <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">Equality Comparable</a>.
 */
template <typename T1, typename T2>
    static
    inline
    bool operator!=(const pair<T1,T2> &x, const pair<T1,T2> &y) restrict(cpu, amp);


/*! This operator tests two pairs for descending ordering.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>y < x</tt>.
 *
 *  \tparam T1 is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 *  \tparam T2 is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 */
template <typename T1, typename T2>
    static
    inline
    bool operator>(const pair<T1,T2> &x, const pair<T1,T2> &y) restrict(cpu, amp);


/*! This operator tests two pairs for ascending ordering or equivalence.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>!(y < x)</tt>.
 *
 *  \tparam T1 is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 *  \tparam T2 is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 */
template <typename T1, typename T2>
    static
    inline
    bool operator<=(const pair<T1,T2> &x, const pair<T1,T2> &y) restrict(cpu, amp);


/*! This operator tests two pairs for descending ordering or equivalence.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>!(x < y)</tt>.
 *
 *  \tparam T1 is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 *  \tparam T2 is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 */
template <typename T1, typename T2>
    static
    inline
    bool operator>=(const pair<T1,T2> &x, const pair<T1,T2> &y) restrict(cpu, amp);



/*! This function creates a \p pair from two objects.
 *
 *  \param x The first object to copy from.
 *  \param y The second object to copy from.
 *  \return A newly-constructed \p pair copied from \p a and \p b.
 *
 *  \tparam T1 There are no requirements on the type of \p T1.
 *  \tparam T2 There are no requirements on the type of \p T2.
 */
template <typename T1, typename T2>
static
inline
pair<T1,T2> make_pair(T1 x, T2 y) restrict(cpu, amp);


/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns either the type of a \p pair's
 *  \c first_type or \c second_type in its nested type, \c type.
 *
 *  \tparam N This parameter selects the member of interest.
 *  \tparam T A \c pair type of interest.
 */
template<int N, typename T> struct tuple_element;


/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns \c 2, the number of elements of a \p pair,
 *  in its nested data member, \c value.
 *
 *  \tparam Pair A \c pair type of interest.
 */
template<typename Pair> struct tuple_size;



/*! \} // pair
 */

/*! \} // Miscellaneous
 */
    } //end cl
} // end bolt

#include <bolt/amp/detail/pair.inl>

#endif
