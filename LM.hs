{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}

import TypedFlow.Layers.RNN.Base
import TypedFlow.TF
import TypedFlow
import TypedFlow.Types
import TypedFlow.Types.Proofs
import GHC.TypeLits
import TypedFlow.Layers.Core (DenseP(..),(#))
import Prelude hiding (RealFrac(..))
import TypedFlow.Python
import OrnUtils
import XOrn (mkXORN)
import FFTOrn (mkFORN, type Pow2, type OrnAnglesShape, type OrnEmbSize, knownPowTwo, knownOrnAnglesShape)
import LSTMUtils
import System.Environment

lstmModel :: forall (len :: Nat) (vocSize::Nat). KnownNat vocSize => KnownNat len =>
   Gen [Function]
lstmModel = do
  -- dropw <- mkDropout (DropProb 0.95)
  embs <- parameterDefault "embs"
  -- wstk <- parameterDefault "stk"
  let dropProb = DropProb 0.10
  -- lstm1 <- mkLSTM @40 "w1" dropProb
  -- gru1 <- mkGRU @160 "w1" dropProb
  lstm1 <- mkLSTM @160 "w1" dropProb
  -- lstm2 <- mkLSTM @160 "w2" dropProb
  -- lstm3 <- mkLSTM @160 "w3" dropProb
  -- lstm4 <- mkLSTM @160 "w4" dropProb
  -- lstm2 <- mkLSTM @160 "w2" dropProb
  drp <- mkDropout dropProb
  w <- parameterDefault "dense"
  let lm :: (Placeholders '[ '("x",'[len],Int32),
                             '("y",'[len],Int32),
                             '("weights",'[len],Float32)]) ->
         ModelOutput Float32 '[len,vocSize] '[]
      lm (PHT  input :* PHT gold :* PHT masks :* Unit) =
        let (_sFi,predictions)
              = simpleRnn (timeDistribute (embedding @12 @vocSize embs) .-.
                  -- gru1 .-.
                  lstm1 .-.
                  -- lstm2.-. lstm3.-. lstm4 .-.
                  -- (stackCell @12 wstk) .-.
                  -- withFeedback (lstm1 .-. (stackCell @8 wstk)) .-.
                  timeDistribute drp .-.
                  timeDistribute (dense w))
                 (repeatT zeros,input)
        in timedCategorical masks predictions gold
  return [modelFunction "runModel" lm]

urnModel :: forall (len :: Nat) (vocSize::Nat) units embSize stateShape.
              (units ~ 32, embSize ~ 90) => -- 3 bands + 2
              -- (units ~ 32, embSize ~ 88) => -- 3 bands
              -- (units ~ 50, embSize ~ (49 + 48 + 47)) => -- 3 bands
              -- (units ~ 50, embSize ~ (25*49)) => -- 3 full
              -- (units ~ 32, l ~ 496) => -- full
              (stateShape ~ '[ '[units]]) =>
              KnownNat vocSize => KnownNat len =>
   Gen [Function]
urnModel = do
  embs@(EmbeddingP embMat) <- parameterDefault "embs" -- embedding layer
  let dropProb = DropProb 0.05
  theRNN <- mkD @embSize @units dropProb
  drp <- mkDropout dropProb
  drp1 <- mkDropout dropProb
  -- w <- parameter "projection" glorotUniform -- projection layer
  proj <- parameterDefault "projection"
  let initialState :: HTV Float32 stateShape
      initialState = VecSing (oneHot0 @units @'B32 (constant 0))    -- start with [1 0 0 0 ... ] vector
      -- projectionLayer = timeDistribute (w ∙)
      projectionLayer = timeDistribute (proj #)
      base = timeDistribute (embedding @embSize @vocSize embs) .-.
             timeDistribute drp1 .-.                       
             theRNN .-.
             timeDistribute drp
      run :: KnownTyp t1 => KnownShape s1 => KnownShape s0 => KnownNat n => (t2 ~ Float32)
          => RnnCell t2 stateShape (T s1 t1) (T s0 t0)
          -> (Tensor (n : s1) t1)
          -> (Tensor (n : s0) t0)
      run cell input = snd $ simpleRnn cell (initialState ,input)
      net = base .-. projectionLayer
      lm :: Placeholders '[ '("x",'[len],Int32), '("y",'[len],Int32), '("weights",'[len],Float32)] -> ModelOutput Float32 '[len,vocSize] '[]
      lm (PHT input :* PHT gold :* PHT masks :* Unit) = timedCategorical masks (run net input) gold
      probeStates :: Placeholders '[ '("x",'[len],Int32) ] -> Placeholders '[ '("states", '[len,units], Float32)  ]
      probeStates (PHT input :* Unit) = PHT ((run base) input) :* Unit
      probeEmbs :: Placeholders '[ '("wordIdx", '[], Int32) ]
                -> Placeholders '[ '("embsAntiHermitian", '[units,units], Float32)  ]
      probeEmbs (PHT idx :* Unit) = PHT (embToAntiHermitian @embSize (lookupT idx embMat)) :* Unit
      probePreds :: Placeholders '[ '("x",'[len],Int32) ] -> Placeholders '[ '("pred" , '[len,vocSize], Float32), '("y" , '[len], Int32)]
      probePreds (PHT input :* Unit) = let ps = run net input
                                       in PHT ps :* PHT (mapT argmax0 ps)  :* Unit
  return [modelFunction "runModel" lm
         ,probeFunction "probeStates" probeStates
         ,probeFunction "probePreds" probePreds
         ,probeFunction "probeEmbs" probeEmbs]


furnModel :: forall (len :: Nat) (vocSize::Nat) units embSize stateShape.
              (stateShape ~ '[ '[ Pow2 units] ]
              ,embSize ~ OrnEmbSize units
              ,KnownPeano units
              -- ,units ~ 'Succ ('Succ ('Succ 'Zero))
              ,units ~ 'Succ ('Succ ('Succ ('Succ ('Succ 'Zero)))) -- attn: 2^...
              ) =>
              KnownNat vocSize => KnownNat len =>
   Gen [Function]
furnModel = knownPowTwo @units ?>
            knownOrnAnglesShape (knownPeano @units) ?>
            knownProduct @(OrnAnglesShape units) ?>
  do
  initState <- normalize <$> parameter "initstate" (noise (UniformD (-1) 1)) -- can't use the 1 0 0 0 state because the network is not isotropic!
  embMat <- parameter "embs" (noise (UniformD (-pi/3) (pi/3)))
  let embs = EmbeddingP embMat --  <- parameterDefault "embs" -- embedding layer
  let dropProb = DropProb 0
  theRNN <- mkFORN  @units dropProb
  drp <- mkDropout dropProb
  drp1 <- mkDropout dropProb
  -- w <- parameter "projection" glorotUniform -- projection layer
  proj <- parameterDefault "projection"
  let initialState :: HTV Float32 stateShape
      initialState = VecSing initState 
      -- projectionLayer = timeDistribute (w ∙)
      projectionLayer = timeDistribute (proj #)
      base = timeDistribute (embedding @(Product (OrnAnglesShape units)) @vocSize embs) .-.
             timeDistribute drp1 .-.                       
             theRNN .-.
             timeDistribute drp
      run :: KnownTyp t1 => KnownShape s1 => KnownShape s0 => KnownNat n => (t2 ~ Float32)
          => RnnCell t2 stateShape (T s1 t1) (T s0 t0)
          -> (Tensor (n : s1) t1)
          -> (Tensor (n : s0) t0)
      run cell input = snd $ simpleRnn cell (initialState ,input)
      net = base .-. projectionLayer
      lm :: Placeholders '[ '("x",'[len],Int32), '("y",'[len],Int32), '("weights",'[len],Float32)] -> ModelOutput Float32 '[len,vocSize] '[]
      lm (PHT input :* PHT gold :* PHT masks :* Unit) = timedCategorical masks (run net input) gold
      -- probeStates :: Placeholders '[ '("x",'[len],Int32) ] -> Placeholders '[ '("states", '[len,units], Float32)  ]
      -- probeStates (PHT input :* Unit) = PHT ((run base) input) :* Unit
      -- probeEmbs :: Placeholders '[ '("wordIdx", '[], Int32) ]
      --           -> Placeholders '[ '("embsAntiHermitian", '[units,units], Float32)  ]
      -- probeEmbs (PHT idx :* Unit) = PHT (embToAntiHermitian @embSize (lookupT idx embMat)) :* Unit
      -- probePreds :: Placeholders '[ '("x",'[len],Int32) ] -> Placeholders '[ '("pred" , '[len,vocSize], Float32), '("y" , '[len], Int32)]
      -- probePreds (PHT input :* Unit) = let ps = run net input
      --                                  in PHT ps :* PHT (mapT argmax0 ps)  :* Unit
  return [modelFunction "runModel" lm
         -- ,probeFunction "probeStates" probeStates
         -- ,probeFunction "probePreds" probePreds
         -- ,probeFunction "probeEmbs" probeEmbs
         ]


xurnModel :: forall (len :: Nat) (vocSize::Nat) n l embShape stateShape.
              (embShape ~  '[ l * n ]
              ,stateShape ~ '[ '[n*2 + 1] ]
              ,KnownNat n
              ,KnownNat l
              -- , n ~ 31
              -- ,l ~ 3
              )
          => KnownNat vocSize => KnownNat len
          => Gen [Function]
xurnModel = 
  do
  embMat <- parameter "embs" (noise (UniformD (-pi/3) (pi/3)))
  let embs = (EmbeddingP @vocSize embMat) --  <- parameterDefault "embs" -- embedding layer
  let dropProb = DropProb 0.05
  (theRNN :: (RnnCell t '[ '[n*2 + 1] ] (Tensor '[l*n] t) (Tensor '[n*2 + 1] t))) <- mkXORN @l @n dropProb
  drp <- mkDropout dropProb
  drp1 <- mkDropout dropProb
  -- w <- parameter "projection" glorotUniform -- projection layer
  proj <- parameterDefault "projection"
  let initialState :: HTV Float32 stateShape 
      initialState = VecSing (oneHot0 @(n*2+1) @'B32 (constant 0))    -- start with [1 0 0 0 ... ] vector
      -- projectionLayer = timeDistribute (w ∙)
      projectionLayer = timeDistribute (proj #)
      base = timeDistribute (embedding @(l*n) @vocSize embs) .-.
             timeDistribute drp1 .-.                       
             theRNN .-.
             timeDistribute drp
      run :: KnownTyp t1 => KnownShape s1 => KnownShape s0 => KnownNat len => (t2 ~ Float32)
          => RnnCell t2 stateShape (T s1 t1) (T s0 t0) -> (Tensor (len : s1) t1) -> (Tensor (len : s0) t0)
      run cell input = snd $ simpleRnn cell (initialState ,input)
      net = base .-. projectionLayer
      lm :: Placeholders '[ '("x",'[len],Int32), '("y",'[len],Int32), '("weights",'[len],Float32)] -> ModelOutput Float32 '[len,vocSize] '[]
      lm (PHT input :* PHT gold :* PHT masks :* Unit) = timedCategorical masks (run net input) gold
      -- probeStates :: Placeholders '[ '("x",'[len],Int32) ] -> Placeholders '[ '("states", '[len,units], Float32)  ]
      -- probeStates (PHT input :* Unit) = PHT ((run base) input) :* Unit
      -- probePreds :: Placeholders '[ '("x",'[len],Int32) ] -> Placeholders '[ '("pred" , '[len,vocSize], Float32), '("y" , '[len], Int32)]
      -- probePreds (PHT input :* Unit) = let ps = run net input
      --                                  in PHT ps :* PHT (mapT argmax0 ps)  :* Unit
  return [modelFunction "runModel" lm
         -- ,probeFunction "probeStates" probeStates
         -- ,probeFunction "probePreds" probePreds
         -- ,probeFunction "probeEmbs" probeEmbs
         ]


mulModel :: forall (len :: Nat) (vocSize::Nat) units stateShape embSize.
              (units ~ 50) =>
              (embSize ~ (units * units)) =>
              -- (units ~ 32, l ~ 496) => -- full
              (stateShape ~ '[ '[units]]) =>
              KnownNat vocSize => KnownNat len =>
   Gen [Function]
mulModel = do
  embs <- parameterDefault "embs" -- embedding layer
  let dropProb = DropProb 0.05
  theRNN <- mkMul @units dropProb
  drp <- mkDropout dropProb
  drp1 <- mkDropout dropProb
  -- w <- parameter "projection" glorotUniform -- projection layer
  proj <- parameterDefault "projection"
  let initialState :: HTV Float32 stateShape
      initialState = VecSing (oneHot0 @units @'B32 (constant 0))    -- start with [1 0 0 0 ... ] vector
      -- projectionLayer = timeDistribute (w ∙)
      projectionLayer = timeDistribute (proj #)
      base = timeDistribute (embedding @embSize @vocSize embs) .-.
             timeDistribute drp1 .-.                       
             theRNN .-.
             timeDistribute drp
      run :: KnownTyp t1 => KnownShape s1 => KnownShape s0 => KnownNat n => (t2 ~ Float32)
          => RnnCell t2 stateShape (T s1 t1) (T s0 t0)
          -> (Tensor (n : s1) t1)
          -> (Tensor (n : s0) t0)
      run cell input = snd $ simpleRnn cell (initialState ,input)
      net = base .-. projectionLayer
      lm :: Placeholders '[ '("x",'[len],Int32), '("y",'[len],Int32), '("weights",'[len],Float32)] -> ModelOutput Float32 '[len,vocSize] '[]
      lm (PHT input :* PHT gold :* PHT masks :* Unit) = timedCategorical masks (run net input) gold
      probeStates :: Placeholders '[ '("x",'[len],Int32) ] -> Placeholders '[ '("states", '[len,units], Float32)  ]
      probeStates (PHT input :* Unit) = PHT ((run base) input) :* Unit
      probePreds :: Placeholders '[ '("x",'[len],Int32) ] -> Placeholders '[ '("pred" , '[len,vocSize], Float32), '("y" , '[len], Int32)]
      probePreds (PHT input :* Unit) = let ps = run net input
                                       in PHT ps :* PHT (mapT argmax0 ps)  :* Unit
  return [modelFunction "runModel" lm
         ,probeFunction "probeStates" probeStates
         ,probeFunction "probePreds" probePreds]



main :: IO ()
main = do
  [modelKind] <- getArgs
  let targetModel :: forall (len :: Nat) (vocSize::Nat). KnownNat len => KnownNat vocSize => Gen [Function]
      targetModel = case modelKind of
                     "forn"  -> furnModel @len @vocSize
                     "orn"   -> urnModel  @len @vocSize
                     "lstm"  -> lstmModel @len @vocSize
                     "mul"   -> mulModel  @len @vocSize
                     "xorn"  -> xurnModel @len @vocSize @32 @5
                     _ -> error "main: unknown modelKind"
  generateFile (modelKind <> "_lm.py") (compileGen @512 defaultOptions (targetModel @21 @12))
  putStrLn "done!"
