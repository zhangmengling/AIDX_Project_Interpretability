dtmc

module train_12313
s:[1..6] init 4;
[]s=1 -> 0.42771084337349397:(s'=2) + 0.25301204819277107:(s'=3) + 0.3192771084337349:(s'=4);
[]s=2 -> 0.7029404372958029:(s'=2) + 0.14174415682332245:(s'=3) + 0.1399849208343805:(s'=4) + 0.008544860517718019:(s'=5) + 0.006785624528776075:(s'=6);
[]s=3 -> 0.1341797361214837:(s'=2) + 0.7152103559870551:(s'=3) + 0.13666915608663183:(s'=4) + 0.005476723923325865:(s'=6) + 0.00846402788150361:(s'=5);
[]s=4 -> 0.1437892095357591:(s'=2) + 0.13500627352572145:(s'=3) + 0.709159347553325:(s'=4) + 0.006775407779171894:(s'=6) + 0.005269761606022585:(s'=5);
[]s=5 -> 1:(s'=5);
[]s=6 -> 1:(s'=6);
endmodule

label "L0"  = s=2;
label "L1"  = s=3;
label "L2"  = s=4;
label "LN"  = s=5;
label "LP"  = s=6;
label "LS"  = s=1;