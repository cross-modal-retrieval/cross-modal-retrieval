function B = compress_img(Img, outFCell, R, inPara)
    Img_code = Img * outFCell{1}/(outFCell{1}' * outFCell{1} + (inPara.gamma/inPara.r(1)) * eye(inPara.bits));
    Img_code = Img_code * R;
    Img_code = bsxfun(@minus, Img_code, median(Img_code, 2));
    Img_code = Img_code>=0;
    B = compactbit(Img_code);
end

