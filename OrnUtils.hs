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
module OrnUtils where

import TypedFlow.Layers.RNN.Base
import TypedFlow.TF
import TypedFlow
import TypedFlow.Types.Proofs
import Prelude hiding (RealFrac(..))

type Matching l n = ((2 * l) <= (n-1) * n) -- eg. if n = 4 we have 1+2+3=6 = 3*4/2 elements

embToUnitary, embToAntiHermitian :: forall l n t. KnownBits t => KnownNat l => KnownNat n => Matching l n => T '[l] ('Typ 'Float t) -> T '[n,n] ('Typ 'Float t)
embToUnitary = expm . embToAntiHermitian

embToAntiHermitian = makeAntisym . embToTri

embToTri :: forall l n t. KnownBits t => KnownNat l => KnownNat n => Matching l n => T '[l] ('Typ 'Float t) -> T '[n,n] ('Typ 'Float t)
embToTri = fillUpperTriangular

makeAntisym :: KnownNat n => KnownFloat t => Tensor '[n,n] t -> Tensor '[n,n] t
makeAntisym a = a - transpose01 a

urn :: forall l n t. KnownNat l => Matching l n => (KnownNat n, KnownFloat t) => 
        RnnCell t '[ '[n] ] (Tensor '[l] t) (Tensor '[n] t)
urn xt = C $ \(VecSing ht1) ->
  let ht = embToUnitary xt ∙ ht1
  in (VecSing ht , ht)

mkD :: ∀ l n. KnownNat l => Matching l n => KnownNat n => DropProb -> Gen (RnnCell Float32 '[ '[n] ] (Tensor '[l] Float32) (Tensor '[n] Float32))
mkD dropProb = do
  rdrp1 <- mkDropout dropProb
  return (onStates (\(VecSing r) -> VecSing (rdrp1 r)) (urn @l @n) )


mulCell :: forall n t. (KnownNat n, KnownFloat t) => 
        RnnCell t '[ '[n] ] (Tensor '[n*n] t) (Tensor '[n] t)
mulCell xt = C $ \(VecSing ht1) ->
  let mat :: T [n,n] t
      mat = knownProduct @'[n,n] ?> reshape xt
      ht = mat ∙ ht1
  in (VecSing ht , ht)

mkMul :: ∀ n. KnownNat n => DropProb -> Gen (RnnCell Float32 '[ '[n] ] (Tensor '[n*n] Float32) (Tensor '[n] Float32))
mkMul dropProb = do
  rdrp1 <- mkDropout dropProb
  return (onStates (\(VecSing r) -> VecSing (rdrp1 r)) (mulCell @n) )
