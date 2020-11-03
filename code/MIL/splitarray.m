function B = splitarray(A, siz)
%SPLITARRAY Split an array into subarrays.
%
%   SPLITARRAY(A, SIZ) splits the array A into subarrays of size SIZ and
%   returns a cell array with all the subarrays.
%
%   See also NUM2CELL, MERGEARRAY.

%   Author:      Peter J. Acklam
%   Time-stamp:  2002-03-03 13:50:35 +0100
%   E-mail:      pjacklam@online.no
%   URL:         http://home.online.no/~pjacklam

error(nargchk(2, 2, nargin));
if length(siz) == 1, siz = [siz siz]; end

Asiz = size(A);                     % size of A
Adim = length(Asiz);                % dimensions in A
Sdim = length(siz);                 % dimensions in each subarray
Bdim = max(Adim, Sdim);             % dimensions in B (output array)
Asiz = [Asiz ones(1,Bdim-Adim)];    % size of A (padded)
Ssiz = [siz  ones(1,Bdim-Sdim)];    % size of each subarray (padded)

Bsiz = Asiz ./ Ssiz;                % size of B (output array)
if any(Bsiz ~= round(Bsiz))
   error('A can not be divided into subarrays as specified.');
end

% A becomes [ Ssiz(1) Asiz(1)/Ssiz(1) Ssiz(2) Asiz(2)/Ssiz(2) ... ].
A = reshape(A, reshape([Ssiz ; Bsiz], [1 2*Bdim]));

% A becomes [ Ssiz(1) Ssiz(2) ... Asiz(1)/Ssiz(1) Asiz(2)/Ssiz(2) ... ].
A = permute(A, [ 1:2:2*Bdim-1 2:2:2*Bdim ]);

% A becomes [prod(Ssiz) prod(Bsiz)].
A = reshape(A, [prod(Ssiz) prod(Bsiz)]);

B = cell(Bsiz);
for i = 1:prod(Bsiz)
   B{i} = reshape(A(:,i), Ssiz);
end
