function assert(pred, str)
% ASSERT Raise an error if the predicate is not true.
% assert(pred, string)

if nargin<2, str = ''; end

if ~pred
	s = sprintf('UGM error: %s', str);
	error(s);
end
end