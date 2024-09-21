function B = CVH_compress1(feaTest_text, model)

B = compress2code(feaTest_text', model.Wy, 'zero', 0);

end
