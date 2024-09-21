function B = CVH_compress(feaTest_vis, model)

B = compress2code(feaTest_vis', model.Wx, 'zero', 0);

end
