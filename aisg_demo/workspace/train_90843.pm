dtmc

module train_90843
s:[1..5] init 1;
[]s=1 -> 0.5264285714285715:(s'=2) + 0.4735714285714286:(s'=3);
[]s=2 -> 0.7259383464465369:(s'=2) + 0.18395459822004384:(s'=3) + 0.0409647878240681:(s'=4) + 0.04914226750935122:(s'=5);
[]s=3 -> 0.18223120962235412:(s'=2) + 0.7256683649351331:(s'=3) + 0.0479804611586743:(s'=4) + 0.044119964283838436:(s'=5);
[]s=4 -> 1:(s'=4);
[]s=5 -> 1:(s'=5);
endmodule

label "0"  = s=2;
label "1"  = s=3;
label "N"  = s=4;
label "P"  = s=5;
label "S"  = s=1;
