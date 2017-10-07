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
## @deftypefn {Function File} {@var{retval} =} convolve (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Saqib Azim <saqib1707@Barkat>
## Created: 2017-05-28

function [retval] = convolve (input_matrix, conv_filter)
  z = (reshape(input_matrix, (20,20)))';
  sz = size(z);
  retval = zeros(size(z,1)-2, size(z,2)-2);
  for j = 2:sz(1,2)-1,
    for i = 2:sz(1,1)-1,
      mat = z(i-1:i+1, j-1:j+1);
      retval(i-1, j-1) = sum(sum(mat.*conv_filter));
      % Rectified Linear Unit
      if (retval(i-1, j-1) < 0),
        retval(i-1, j-1) = 0;
      end;
    end;
  end;
endfunction
