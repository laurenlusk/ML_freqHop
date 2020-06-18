% Parameters
numChan = 79;     % number of freq channels
startFreq = 2.402e9;    % lowest center freq for channels
endFreq = 2.480e9;    % highest center freq for channels
chanBW = 1e6;   %channel bandwidth is 1MHz, sampling rate 1us

hop = randperm(numChan);  % generates freq hopping pattern
sampPerHop = 625;  % there are 625us/hop, 1600 hops/s

occ = zeros([numChan sampPerHop*numChan]);

for i=1:numChan     % takes channel # & makes it 1 (for occupied)
    for j=1:sampPerHop
        occ(hop(i),j+(sampPerHop*(i-1))) = 1;    % changes occupied channel's status to occupied (1)
    end
end

occ = occ';
occ = [occ; occ];
csvwrite("all_data2.csv",occ);