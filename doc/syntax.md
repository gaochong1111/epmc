# 符号说明
{}: 0到多次  
[]: 0或1次  
{}+: 1或多次  

# 语法描述
**ModelPRISM**:  
{    
     ModelType  
    |Constant  
    |Label  
    |Global  
    |Formula  
    |Module  
    |Rewards  
    |Init  
    |System   
}  

**ModelType**:   
"mdp"   
|"nondeterministic"   
|"dtmc"   
|"probabilistic"   
|"qmc"   
|"ctmc"   
|"stochastic"   
|"pta"   

**Constant**:  
 ConstantQmc   
 | ( "const"   
     (  ["int"|"double"|"bool"] 
     <b>|</b>
         ("rate"|"prob") )   
     Identifier    
     (
         ("=" ExpSemicolon ";") 
     <b>|</b> ";" ) 
   )  

 *注意*: 这里的 ["int"|"double"|"bool"] 是可以缺省的, 默认为int，比如 const Identifier = Exp;是合法的。  
 ***修改意见***: 修改不要可缺省。  
 修改为：ConstantQmc | ( "const" ("int"|"double"|"bool"|"rate"|"prob") ) Identifier (("=" ExpSemicolon ";") | ";") )
 
 **ConstantQmc**:  
 "const"   
 (
     ("vector" ( ("<" Identifier "|") | ("|" Identifier ">") ) "_" IdOrInt )   
     | ( "matrix" Identifier )   
     | ( "superoperator(" Int ")" Identifier ))  
     "=" ExpSemicolon ";"  

***修改意见***: vector类型的size不使用下划线表示，矩阵要显示表示size  
修改为：   
"const" (("vector(" Int ")" ( ("<" Identifier "|")|("|" Identifier ">") ))    
| ( "matrix(" Int "," Int ")" Identifier )    
| ( "superoperator(" Int ")" Identifier )) "=" ExpSemicolon ";"
 
 **Label**:  
 "label" "\"" Identifier "\"" "=" ExpSemicolon ";"
 
 **Global**:  
 "global" VariableDeclaration
 
 **Formula**:  
 "formula" Identifier "=" ExpSemicolon ";"
 
 **Module**:  
 "module" Identifier ( ( "=" ModuleRename ) | ModuleContent ) "endmodule"
 
 **ModuleRename**:  
 Identifier "[" Renaming {"," Renaming} "]"
 
 **Renaming**:  
 Identifier "=" Identifier
 
 **ModuleContent**:  
 ({VariableDeclaration} [InvariantDeclaration] {GuardedCommandDeclaration})
 
 **VariableDelaration**:  
 Identifier ":" Variable ( ("init" ExpSemicolon ";") | (";") )
 
 **Variable**:  
 ( "bool" | ("[" ExpSepinterval ".." ExpBrack "]") | "clock" )
 
 **InvariantDeclaration**:  
 "invariant" ExpInvariant "endinvariant"
 
 **GuardedCommandDeclaration**:  
 "[" [Identifier] "]" Condition "->" Update ";"
 
 **Update**:  
 DetUpdate | ProbUpdate
 
 **ProbUpdate**:  
 ( "~[]" | "(" | Identifier | "true") ExpColon ":" DetUpdate { "+" ExpColon ":" DetUpdate}
 
 **DetUpdate**:  
 ( ("true") | ( "(" Identifier "'=" ExpParent ")" { "&(" Identifier "'=" ExpParent ")"}) )
 
 **Rewards**:  
 "rewards" ["\"" Identifier "\""] [ "[" [Identifier] "]" ] {["true"|"false"|Identifier|"("|"-"|"!"] ExpColon ":" ExpSemicolon ";"}+ "endrewards"
 
 **Init**:  
 "init" ExpInit "endinit"
 
 **System**:  
 "system" SystemContent "endsystem"
 
 **SystemContent**:  
 SystemParallelCommonActions
 
 **SystemParallelCommonActions**:  
 SystemParallelAsynchronous {"||" SystemParallelSetActions}
 
 **SystemParallelAsynchronou**:  
 SystemParallelSetActions {"|||" SystemParallelCommonActions}
 
 **SystemParallelSetActions**:  
 SystemRenHid [ "|[" IdSet "]|" SystemRenHid ]
 
 **SystemRenHid**:  
 SystemBase { ( "/{" IdSet "}" ) | ( "{" RenameMap "}" ) }
 
 **RenameMap**:  
 ExpressionIdentifier <- ExpressionIdentifier { "," ExpressionIdentifier "<-" ExpressionIdentifier }
 
 **IdSet**:  
 ExpressionIdentifier { "," ExpressionIdentifier }
 
 **SystemBase**:  
 ( ExpressionIdentifier ) | ( "(" SystemParallelCommonActions ")")

# exp syntax
**Exp**:    
    ExpressionITE

**CompleteProp**:  
(label ":")? Exp (";")?

**ExpTemporal**:     
TemporalBinary

**ExpressionITE**:   
    ExpressionImplies ("?" ExpressionImplies ":" ExpressionITE)?

**ExpressionImplies**:   
    ExpressionIff ("->" ExpressionIff)*

**ExpressionIff**:   
    ExpressionOr ( "<=>" ExpressionOr)*

**ExpressionOr**:   
    ExpressionAnd ( "|" ExpressionAnd)*

**ExpressionAnd**:   
    ExpressionNot ( "&" ExpressionNot)*

**ExpressionNot**:   
    "!" ExpressionNot  
    | ExpressionEqNe

**ExpressionEqNe**:     
    ExpressionROp (EqNe ExpressionROp)*

**EqNe**: "=" | "!="

**ExpressionROp**:   
    ExpressionPlusMinus (LtGeLeGe ExpressionPlusMinus)*

**LtGeLeGe**: "<" | "<=" | ">" | ">="

**ExpressionPlusMinus**:   
    ExpressionTimesDivide (PlusMinus ExpressionTimesDivide)*

**PlusMinus**: "+" | "-"

**ExpressionTimesDivide**:  
    ExpressionUnaryMinus ( TimesDivide ExpressionUnaryMinus)*

**TimesDivide**: "*" | "/"

**ExpressionUnaryMinus**:
    "-" ExpressionUnaryMinus   
    | ExpressionTranspose

**ExpressionTranspose**:   
    Basic ("'")?

**Basic**:   
    Imaginary
    | Boolean
    | Function
    | Identifier
    | Int
    | Real
    | Matrix
    | ProbQuant
    | RewQuant
    | SteadyQuant
    | SuperOperator
    | Parenth
    | Label
    | Filter
    | BraKet

**BraKet**:
    ("<" IdOrInt "|" (IdOrInt ">")? "_" IdOrInt     
    | "|" IdOrInt ">" "_" IdOrInt)  
    ("<" IdOrInt "|" "_" IdOrInt    
    | "|" IdOrInt ">" "_" IdOrInt)?  

**IdOrInt**: Int | Identifier

**Function**:  
    SpecialFunction
    | FunctionMultipleParams "(" Function2 ")"  
    | FunctionOneParam "(" Function1 ")"  
    | "func" "(" 
        ( Sqrt 
        | FunctionMultipleParams "," FunctionN
        | FunctionTwoParams "," Function2
        | FunctionOneParam "," Function1 ) 
        ")"

**SpecialFunction**:   
     Sqrt | Ctran

**Sqrt**: "sqrt" "(" Exp ")"

**Ctran**: "ctran" "(" Exp ")"

**FunctionN**: Exp ("," Exp)+

**Function2**: Exp "," Exp

**Function1**: Exp

**FunctionMultipleParams**: "max" | "min"

**FunctionOneParam**: "floor" | "ceil" | "tran" | "conj"

**FunctionTwoParams**: "pow" | "mod" | "log" | "qeval" | "qprob"

**Parenth**: "(" ExpTemporal ")"

**Identifier**: ID

**Label**: QUOTE (Identifier | "init") QUOTE

**Real**: NUM_REAL

**Imaginary**: IMAG
    | NUM_REAL IMAG

**Int**: NUM_INT

**SuperOperator**: SUPERATOR_OPEN ExpList SUPERATOR_CLOSE
    | "mf2so" "(" Exp ")"

**ExpList**: Exp ( "," Exp)*

**Vector**: "{" ExpList "}"

**Boolean**: "true" | "false"

**Matrix**: ("Identity" | "ID") "(" Exp ")"
    | ( "Paulix" | "PX")
    | ( "Pauliy" | "PY")
    | ( "Pauliz" | "PZ")
    | ( "Hadamard" | "HD")
    | ( "CNOT" | "CN")
    | "M0"
    | "M1"
    | ("PhaserShift" | "PS") "(" Exp ")"
    | ( "Swap" | "SW")
    | ( "Toffoli" | "TF")
    | ( "Fredkin" | "FK")
    | SingleMatrix

**SingleMatrix**: "{" MatrixRow (";" MatrixRow)* "}"

**MatrixRow**: Exp ( "," Exp)*

**OldSchoolFilter**: "{" ExpTemporal "}" 
    ("{" ( "max" "}" | "min" "}" ("{" "max" "}")?) )?

**ProbQuant**:
    PropQuantProbDirType 
    (
        ("=" ("?" | ExpTemporal) "[" ExpTemporal ("given" ExpTemporal)? (OldSchoolFilter)? "]")
    | PropQuantCmpType ExpTemporal "[" ExpTemporal ("given" ExpTemporal)? (OldSchoolFilter)? "]"
    )

**ProbQuant**:
    SteadyQuantProbDirType 
    (
        ("=" ("?" | ExpTemporal) "[" ExpTemporal ("given" ExpTemporal)? (OldSchoolFilter)? "]")
    | PropQuantCmpType ExpTemporal "[" ExpTemporal ("given" ExpTemporal)? (OldSchoolFilter)? "]"
    )

**PropQuantCmpType**: "<=" | "<" | ">=" | ">"

**RewardPath**: 
    "F" ExpTemporal
    | "C" ("<=" ExpTemporal)? ("," "DISCOUNT" "=" ExpTemporal)?
    | "I" "=" ExpTemporal
    | "S"

**RewardStructure**: 
    "{" (QUOTE IDENTIFIER QUOTE | Exp) "}"

**RewQuant**: 
    (
        "R" (RewardStructure)? ("min" | "max")?
        | "Rmin"
        | "Rmax"
    )
    (
        "=" ("?" | ExpTemporal)
        | PropQuantCmpType ExpTemporal
    )
    "[" RewardPath ("given" ExpTemporal)? (OldSchoolFilter)? "]"

**PropQuantProbDirType**: "P" | "Pmax" | "Pmin" | "Q" | "Qmax" | "Qmin"

**SteadyQuantProbDirType**: "S" | "Smax" | "Smin"

**ExpressionFilterType**: "min" | "max" | "+" | "&" | "|" | IDENTIFIER

    -支持-: count | sum | avg | first | range | forall | exists | state | argmin | argmax | print | printall

**Filter**: 
    "filter" "(" ExpressionFilterType "," ExpTemporal ( "," ExpTemporal)? ")"

**TimeBound**: 
    (
        "<" Exp
        | "<=" Exp
        | ">" Exp
        | ">=" Exp
        | ("["|"]") Exp "," Exp ("]" | "[")
    )?

**TemporalBinary**: 
    TemporalUnary (TempBinType TimeBound "(" TemporalBinary ")")?

**TempBinType**: "W" | "R" | "U"

**TemporalUnary**: 
    TempUnType TimeBound "(" TemporalUnary ")"
    | ExpressionITE

**TempUnType**: "X" | "F" | "G"




 # 结束语
 - 按照prism中5个运算符的结合方式及优先级重新核对语法 http://www.prismmodelchecker.org/manual/ThePRISMLanguage/AllOnOnePage
 
 
 
