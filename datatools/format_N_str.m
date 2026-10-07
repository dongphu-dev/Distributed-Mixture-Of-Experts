function s = format_N_str(n)
%% FORMAT_N_STR
% Standard utility to format sample sizes into clean, scientific string identifiers:
%   100000  -> '100k'
%   300000  -> '300k'
%   1000000 -> '1M'
%   otherwise -> string representation

    if n >= 1e6 && mod(n, 1e6) == 0
        s = sprintf('%dM', round(n / 1e6));
    elseif n >= 1e3 && mod(n, 1e3) == 0
        s = sprintf('%dk', round(n / 1e3));
    else
        s = num2str(n);
    end
end
