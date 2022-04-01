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
module OrnClassifier where

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
import LSTMUtils

predictLSTM :: forall (len :: Nat) (vocSize::Nat) units embSize.
  (units ~ 32, embSize ~ 150) =>
  (KnownNat vocSize, KnownNat len) =>
  Gen [Function]
predictLSTM = do
  embs <- parameterDefault "embs"
  let dropProb = DropProb 0.1
  lstm1 <- mkLSTM @units "w1" dropProb
  drp <- mkDropout dropProb
  drp1 <- mkDropout dropProb
  w <- parameterDefault "dense" -- projection layer
  let predictor :: Placeholders '[ '("x",'[len],Int32), '("yIndex",'[],Int32), '("y",'[],Int32)]
                -> ModelOutput Float32 '[2] '[] 
      predictor (PHT input :* PHT yIndex :* PHT y :* Unit) = 
        let (_sFi,predictions) = simpleRnn (timeDistribute (embedding @embSize @vocSize embs) .-.
                                            timeDistribute drp1 .-.                       
                                            lstm1 .-.
                                            timeDistribute drp .-.
                                            timeDistribute (dense w))
              (repeatT zeros,input) -- start with 0 vector
        in categoricalDistribution (lookupT yIndex predictions) (oneHot0 @2 y)
  return [modelFunction "runModel" predictor]


-- >>> 47 + 48 + 49
-- 144

predictOrn :: forall (len :: Nat) (vocSize::Nat) units embSize stateShape.
  (units ~ 50, embSize ~ 144,
   stateShape ~ '[ '[units]]) =>
  (KnownNat vocSize, KnownNat len) =>
  Gen [Function]
predictOrn = do
  embs@(EmbeddingP embMat) <- parameterDefault "embs"
  let dropProb = DropProb 0.05
  theRNN <- mkD @embSize @units dropProb
  drp <- mkDropout dropProb
  drp1 <- mkDropout dropProb
  w <- parameterDefault "dense" -- projection layer
  let initialState :: HTV Float32 stateShape
      initialState = VecSing (oneHot0 @units @'B32 (constant 0))    -- start with [1 0 0 0 ... ] vector
      run :: KnownTyp t1 => KnownShape s1 => KnownShape s0 => KnownNat n => (t2 ~ Float32)
          => RnnCell t2 stateShape (T s1 t1) (T s0 t0)
          -> (Tensor (n : s1) t1)
          -> (Tensor (n : s0) t0)
      run cell input = snd $ simpleRnn cell (initialState ,input)
      predictor :: Placeholders '[ '("x",'[len],Int32), '("yIndex",'[],Int32), '("y",'[],Int32)]
                -> ModelOutput Float32 '[2] '[]
      net = timeDistribute (embedding @embSize @vocSize embs) .-.
            timeDistribute drp1 .-.                       
            theRNN .-.
            timeDistribute drp .-.
            timeDistribute (dense w)
      probePreds :: Placeholders '[ '("x",'[len],Int32) ] -> Placeholders '[ '("pred" , '[len,2], Float32), '("y" , '[len], Int32)]
      probePreds (PHT input :* Unit) = let ps = (run net input)
                                       in PHT ps :* PHT (mapT argmax0 ps)  :* Unit
      predictor (PHT input :* PHT yIndex :* PHT y :* Unit) = 
        categoricalDistribution (lookupT yIndex (run net input)) (oneHot0 @2 y)
      probeEmbs :: Placeholders '[ '("wordIdx", '[], Int32) ]
                -> Placeholders '[ '("embsAntiHermitian", '[units,units], Float32)  ]
      probeEmbs (PHT idx :* Unit) = PHT (embToAntiHermitian @embSize (lookupT idx embMat)) :* Unit
  return [modelFunction "runModel" predictor
         ,probeFunction "probeEmbs" probeEmbs
         ,probeFunction "probePreds" probePreds
         ]

main :: IO ()
main = do
  generateFile "orn_classifier.py"  (compileGen @512 defaultOptions (predictOrn @50 @50050))
  generateFile "lstm_classifier.py" (compileGen @512 defaultOptions (predictLSTM @50 @50050))
  putStrLn "done!"
