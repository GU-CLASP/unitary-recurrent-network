{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE GADTs #-}
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


---------------------
-- Li-style

module XOrn where

import TypedFlow.TF
import TypedFlow
import TypedFlow.Types.Proofs
import Prelude hiding (RealFrac(..))

pack2 :: forall s t. KnownShape s => KnownTyp t => Tensor s t -> Tensor s t -> Tensor (2 ': s) t
pack2 x y = stack0 (x :** y :** VUnit)

sPeanoKnownNat :: SPeano n -> Sat KnownNat (PeanoNat n)
sPeanoKnownNat SZero = Sat
sPeanoKnownNat (SSucc n) = knownSumS (sPeanoKnownNat n :* Sat :* Unit)
  -- sPeanoKnownNat n ?> knownSat


peanoNatSucc :: forall m n. (n ~ 'Succ m) => SPeano m -> (0 :<: PeanoNat n)
peanoNatSucc _ = plusComm @1 @(PeanoNat m) #>
                 succPos @(PeanoNat m)

applyRot2 :: KnownFloat t => Scalar t -> Tensor '[2] t -> Tensor '[2] t
applyRot2 angle x = m ∙ x
  where s = sin angle
        c = cos angle
        m = pack2 (pack2 c (negate s)) (pack2 s c)

xrotH :: forall n t. (KnownNat n, KnownFloat t) => T '[n] t -> T '[n*2] t -> T '[n*2] t
xrotH angles u  = knownProduct @'[n,2] ?>
                  reshape @'[n*2] ( zipWithT applyRot2 angles (reshape @'[n,2] u))

xrot0 :: forall n t. (KnownNat n, KnownFloat t) => T '[n] t -> T '[n*2+1] t -> T '[n*2+1] t
xrot0 angles x  = knownProduct @'[n,2] ?>
                  consT0 (headT0 x) (xrotH angles (tailT0 x))

xrot1 :: forall n t. (KnownNat n, KnownFloat t) => T '[n] t -> T '[n*2+1] t -> T '[n*2+1] t
xrot1 angles x  = knownProduct @'[n,2] ?>
                  knownPlus @(n*2) @1 ?>
                  snocT0  (xrotH angles (initT0 x)) (last0 x)

xRot' :: (KnownNat n, KnownFloat t) => Bool -> V l (T '[n] t) -> T '[n*2+1] t -> T '[n*2+1] t
xRot' _ VUnit x = x
xRot' which (a :** as) x = (if which then xrot0 else xrot1) a (xRot' (not which) as x)

xRot :: (KnownFloat t, KnownNat n, KnownNat l)
     => Tensor '[l, n] t -> T '[n*2 + 1] t -> T '[n*2 + 1] t
xRot as x = xRot' True (unstack0 as) x


  

xorn :: forall l n t. (KnownNat n, KnownNat l, KnownFloat t) => 
        RnnCell t '[ '[n*2 + 1] ] (Tensor '[l*n] t) (Tensor '[n*2 + 1] t)
xorn xt = C $ \(VecSing ht1) ->
                knownProduct @[l,n] ?>
                let ht = xRot (reshape @[l,n] xt) (ht1)
                in (VecSing ht , ht)

mkXORN :: ∀ l n t. (KnownNat n, KnownNat l, KnownFloat t)
       => DropProb
       -> Gen (RnnCell t '[ '[n*2 + 1] ] (Tensor '[l*n] t) (Tensor '[n*2 + 1] t))
mkXORN dropProb = knownProduct @'[n,2] ?>
                  knownPlus @(n*2) @1 ?>
  do rdrp1 <- mkDropout dropProb
     return (onStates (\(VecSing r) -> VecSing (rdrp1 r)) (xorn @l @n) )
