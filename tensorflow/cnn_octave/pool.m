## Copyright (C) 2017 Saqib Azim
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} pool (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Saqib Azim <saqib1707@Barkat>
## Created: 2017-05-28

function [retval] = pool (input_matrix)
  
  sz = size(input_matrix);
  retval = zeros(floor(sz(1,1)/2), floor(sz(1,2)/2));
  for j = 1:2:sz(1,2),
    for i = 1:2:sz(1,1),
      retval(floor(i/2)+1,floor(j/2)+1) = max(max(input_matrix(i:i+1, j:j+1)));
      
endfunction
