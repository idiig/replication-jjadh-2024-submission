echo "Translator,PoemID,opN,ctN,TotalMatch,Unmatch,ExactMatch,FieldMatch,GroupMatch,AdditionRate,UnmatchRate,TotalMatch_a,Unmatch_a,ExactMatch_a,FieldMatch_a,GroupMatch_a,AdditionRate_a,UnmatchRate_a" > ./artifacts/calc_results.csv
{ tail -n +2 -q ./artifacts/calc_results/*.csv; } >> ./artifacts/calc_results.csv
