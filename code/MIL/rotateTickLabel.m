%%  TH=ROTATETICKLABEL(H,ROT) is the calling form where H is a handle to
%   the axis that contains the XTickLabels that are to be rotated. ROT is
%   an optional parameter that specifies the angle of rotation. The default
%   angle is 90. TH is a handle to the text objects created. For long
%   strings such as those produced by datetick, you may have to adjust the
%   position of the axes so the labels don't get cut off.
%   Of course, GCA can be substituted for H if desired.
function th=rotateTickLabel(h,rot,type)
    %set the default rotation if user doesn't specify
    if nargin==1
        rot=90;
    elseif nargin==2
        type = 1;
    end
    %make sure the rotation is in the range 0:360 (brute force method)
    while rot>360
        rot=rot-360;
    end
    while rot<0
        rot=rot+360;
    end
    %get current tick labels
    a=get(h,'XTickLabel');
    %erase current tick labels from figure
    set(h,'XTickLabel',[]);
    %get tick label positions
    b=get(h,'XTick');
    c=get(h,'YTick');
    %make new tick labels
    if rot<180
        if type
            th=text(b,repmat(c(1)-.1*(c(2)-c(1)),length(b),1),a,'HorizontalAlignment','right','rotation',rot, 'FontSize', 16, 'FontWeight', 'b', 'Fontname','Times','color','k');
        else
            th=text(b,repmat(c(end)+(c(1)-.5*(c(2)-c(1)))*.7,length(b),1),a,'HorizontalAlignment','center','rotation',rot, 'FontSize', 16, 'FontWeight', 'b', 'Fontname','Times','color','k');
        end
    else
        th=text(b,repmat(c(1)-.1*(c(2)-c(1)),length(b),1),a,'HorizontalAlignment','left','rotation',rot,'FontSize', 17, 'Fontname','Times');
    end
    set(gca,'XColor','k');
end