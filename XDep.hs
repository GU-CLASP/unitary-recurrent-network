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
import OrnUtils
import Prelude hiding (RealFrac(..))
import TypedFlow.Python
import LSTMUtils

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
              = simpleRnn (timeDistribute (embedding @20 @vocSize embs) .-.
                  -- gru1 .-.
                  lstm1 .-.
                  -- lstm2.-. lstm3.-. lstm4 .-.
                  timeDistribute drp .-.
                  timeDistribute (dense w))
                 (repeatT zeros,input)
        in timedCategorical masks predictions gold
  return [modelFunction "runModel" lm]

urnModel :: forall (len :: Nat) (vocSize::Nat) units embSize stateShape.
              -- (units ~ 32, embSize ~ 90) => -- 3 bands + 2
              -- (units ~ 32, embSize ~ 88) => -- 3 bands
              -- (units ~ 50, embSize ~ (49 + 48 + 47)) => -- 3 bands
              (units ~ 50, embSize ~ (25*49)) => -- 3 full
              -- (units ~ 32, l ~ 496) => -- full
              (stateShape ~ '[ '[units]]) =>
              KnownNat vocSize => KnownNat len =>
   Gen [Function]
urnModel = do
  embs@(EmbeddingP embMat) <- parameterDefault "embs" -- embedding layer
  let dropProb = DropProb 0.05
  theRNN <- mkD @embSize @units (DropProb 0)
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


main :: IO ()
main = do
  generateFile "orn_xdep.py" (compileGen @512 defaultOptions (urnModel @21 @10))
  generateFile "lstm_xdep.py" (compileGen @512 defaultOptions (lstmModel @21 @10))
  putStrLn "done!"
