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

module LSTMUtils where

import TypedFlow
import TypedFlow.Python

onFST :: (Tensor s1 t -> Tensor s t) -> HTV t '[s1, s'] -> HTV t '[s, s']
onFST f (VecPair h c) = (VecPair (f h) c)

mkLSTM :: ∀ n x. KnownNat x => KnownNat n =>
        String -> DropProb -> Gen (RnnCell Float32 '[ '[n], '[n]] (Tensor '[x] Float32) (Tensor '[n] Float32))
mkLSTM pName dropProb = do
  params <- parameterDefault pName
  drp1 <- mkDropout dropProb
  rdrp1 <- mkDropout dropProb
  return (timeDistribute drp1
          .-.
          onStates (onFST rdrp1) (lstm params))

mkGRU :: ∀ n x. KnownNat x => KnownNat n =>
        String -> DropProb -> Gen (RnnCell Float32 '[ '[n] ] (Tensor '[x] Float32) (Tensor '[n] Float32))
mkGRU pName dropProb = do
  params <- parameterDefault pName
  drp1 <- mkDropout dropProb
  rdrp1 <- mkDropouts dropProb
  return (timeDistribute drp1 .-. onStates rdrp1 (gru params))
