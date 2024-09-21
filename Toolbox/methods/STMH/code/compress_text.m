function B = compress_text(Text, outFCell, numzeros)
    text_code  = multi_STMH_coding(Text', outFCell{2}, numzeros);
    B = compactbit(text_code);
end

