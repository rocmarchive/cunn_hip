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

#if !defined( BOLT_CL_PAIR_INL )
#define BOLT_CL_PAIR_INL
#pragma once


namespace bolt
{
    namespace cl
    {
        //Empty
        template <typename T1, typename T2>
        pair<T1,T2>::pair(void):first(),second()
        {
        } // end pair::pair()

        template <typename T1, typename T2>
        pair<T1,T2>::pair(const T1 &x, const T2 &y):first(x),second(y)
        {
        } // end pair::pair()


        template <typename T1, typename T2>
        template <typename U1, typename U2>
        pair<T1,T2>::pair(const pair<U1,U2> &p):first(p.first),second(p.second)
        {
        } // end pair::pair()


        template <typename T1, typename T2>
        template <typename U1, typename U2>
        pair<T1,T2>::pair(const std::pair<U1,U2> &p):first(p.first),second(p.second)
        {
        } // end pair::pair()

        //todo: Add swap when the implementation is complete

        template <typename T1, typename T2>
        bool operator==(const pair<T1,T2> &x, const pair<T1,T2> &y)
        {
            return x.first == y.first && x.second == y.second;
        } // end operator==()


        template <typename T1, typename T2>
        bool operator<(const pair<T1,T2> &x, const pair<T1,T2> &y)
        {
          return x.first < y.first || (!(y.first < x.first) && x.second < y.second);
        } // end operator<()


        template <typename T1, typename T2>
        bool operator!=(const pair<T1,T2> &x, const pair<T1,T2> &y)
        {
          return !(x == y);
        } // end operator==()


        template <typename T1, typename T2>
        bool operator>(const pair<T1,T2> &x, const pair<T1,T2> &y)
        {
          return y < x;
        } // end operator<()


        template <typename T1, typename T2>
        bool operator<=(const pair<T1,T2> &x, const pair<T1,T2> &y)
        {
          return !(y < x);
        } // end operator<=()


        template <typename T1, typename T2>
        bool operator>=(const pair<T1,T2> &x, const pair<T1,T2> &y)
        {
          return !(x < y);
        } // end operator>=()

        template <typename T1, typename T2>
        pair<T1,T2> make_pair(T1 x, T2 y)
        {
            return pair<T1,T2>(x,y);
        } // end make_pair()


        // specializations of tuple_element for pair
        template<typename T1, typename T2>
          struct tuple_element<0, pair<T1,T2> >
        {
          typedef T1 type;
        }; // end tuple_element

        template<typename T1, typename T2>
          struct tuple_element<1, pair<T1,T2> >
        {
          typedef T2 type;
        }; // end tuple_element


        // specialization of tuple_size for pair
        template<typename T1, typename T2>
          struct tuple_size< pair<T1,T2 > >
        {
          static const unsigned int value = 2;
        }; // end tuple_size



 namespace detail
 {


        template<int N, typename Pair> struct pair_get {};

        template<typename Pair>
          struct pair_get<0, Pair>
        {
            const typename tuple_element<0, Pair>::type &
              operator()(const Pair &p) const
          {
            return p.first;
          } // end operator()()

            typename tuple_element<0, Pair>::type &
              operator()(Pair &p) const
          {
            return p.first;
          } // end operator()()
        }; // end pair_get


        template<typename Pair>
          struct pair_get<1, Pair>
        {
            const typename tuple_element<1, Pair>::type &
              operator()(const Pair &p) const
          {
            return p.second;
          } // end operator()()

            typename tuple_element<1, Pair>::type &
              operator()(Pair &p) const
          {
            return p.second;
          } // end operator()()
        }; // end pair_get

} // end detail



        template<unsigned int N, typename T1, typename T2>
            typename tuple_element<N, pair<T1,T2> >::type &
                get(pair<T1,T2> &p)
        {
            return detail::pair_get<N, pair<T1,T2> >()(p);
        } // end get()

        template<unsigned int N, typename T1, typename T2>
            const typename tuple_element<N, pair<T1,T2> >::type &
                get(const pair<T1,T2> &p)
        {
            return detail::pair_get<N, pair<T1,T2> >()(p);

        } // end get()
    } //end of cl
} // end bolt

#endif


