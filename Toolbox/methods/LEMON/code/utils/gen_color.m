function color=gen_color(curve_idx)

colors=[];

colors = {'m','c','k','g',[0.3 0.8 0.9],'b',[1 0.5 0],[0.5 0.5 0.5],'r',[0.75 0.2 0],...
    [0.7 0 0.7],[0 0.7 0.7],}; %OUR 'r'

sel_idx=mod(curve_idx-1, length(colors))+1;
color=colors{sel_idx};

end
