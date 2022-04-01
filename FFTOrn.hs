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
-- FFT-style 

module FFTOrn where

import TypedFlow.TF
import TypedFlow
import TypedFlow.Types.Proofs
import Prelude hiding (RealFrac(..))
import Data.Type.Equality

type family TwoShape n :: Shape where
  TwoShape 'Zero = '[]
  TwoShape ('Succ n) = 2 ': TwoShape n

type Pow2 n = Product (TwoShape n)

type TwoTensor n t = Tensor (TwoShape n) t

knownTwo :: SPeano n -> Sat KnownShape (TwoShape n)
knownTwo SZero = Sat
knownTwo (SSucc n) = knownTwo n ?> Sat

knownPowTwoS :: forall n. SPeano n -> Sat KnownNat (Pow2 n)
knownPowTwoS n = knownTwo n ?> knownProduct @(TwoShape n)

knownPowTwo :: forall n. KnownPeano n => Sat KnownNat (Pow2 n)
knownPowTwo = knownPowTwoS (knownPeano @n)

pack2 :: forall s t. KnownShape s => KnownTyp t => Tensor s t -> Tensor s t -> Tensor (2 ': s) t
pack2 x y = concat0 x' y'
  where x',y' :: Tensor (1 ': s) t
        x' = expandDim0 x
        y' = expandDim0 y



prf :: SPeano n -> (TwoShape n ++ '[2]) :~: (2 ':TwoShape n)
prf SZero = Refl
prf (SSucc n) = prf n #> Refl

type family OrnAnglesShape (n ::Peano) :: Shape where
  OrnAnglesShape ('Succ n) = (PeanoNat n + 1) ': TwoShape n
  OrnAnglesShape 'Zero = '[] -- in fact, does not exist ()

knownOrnAnglesShape :: SPeano n -> Sat KnownShape (OrnAnglesShape n)
knownOrnAnglesShape SZero = Sat
knownOrnAnglesShape (SSucc n) = sPeanoKnownNat (SSucc n) ?> knownTwo n ?> Sat 
  
type OrnAngles n t = Tensor (OrnAnglesShape n) t

type OrnEmbSize n = Product (OrnAnglesShape n)

knownOrnShape :: SPeano n -> Sat KnownShape (OrnAnglesShape n)
knownOrnShape SZero = Sat
knownOrnShape (SSucc n) = sPeanoKnownNat (SSucc n) ?> knownTwo n ?> Sat

sPeanoKnownNat :: SPeano n -> Sat KnownNat (PeanoNat n)
sPeanoKnownNat SZero = Sat
sPeanoKnownNat (SSucc n) = knownSumS (sPeanoKnownNat n :* Sat :* Unit)
  -- sPeanoKnownNat n ?> knownSat


splitParams' :: forall n t. KnownNumeric t => SPeano n -> OrnAngles ('Succ n) t -> (TwoTensor n t, Tensor (2 ': OrnAnglesShape n) t)
splitParams' SZero prms = (reshape prms,zeros)
splitParams' (SSucc n) prms = knownOrnAnglesShape n ?> knownTwo n ?> sPeanoKnownNat (SSucc n) ?>
   let  h :: TwoTensor n t
        h = headT0 prms
        t :: Tensor (PeanoNat n ': TwoShape n) t
        t =  tailT0 prms
   in  ( h , transpose01 t )

peanoNatSucc :: forall m n. (n ~ 'Succ m) => SPeano m -> (0 :<: PeanoNat n)
peanoNatSucc _ = plusComm @1 @(PeanoNat m) #>
                 succPos @(PeanoNat m)

applyRot2 :: KnownFloat t => Scalar t -> Tensor '[2] t -> Tensor '[2] t
applyRot2 angle x = m ∙ x
  where s = sin angle
        c = cos angle
        m = pack2 (pack2 c (negate s)) (pack2 s c)

map2Sh :: forall n t. KnownFloat t => SPeano n -> TwoTensor n t -> Tensor (TwoShape n ++ '[2]) t -> Tensor (TwoShape n ++ '[2]) t
map2Sh n angles x = knownTwo n ?>
                    appRUnit @(TwoShape n) #>
                    zipWithTT @(TwoShape n) applyRot2 angles x 

applyForn :: forall n t. KnownFloat t => SPeano n -> OrnAngles n t -> TwoTensor n t -> TwoTensor n t
applyForn SZero _ x = x
applyForn (SSucc n) prms x =
  prf n #>
  knownTwo n ?>
  knownOrnShape n ?> 
  (zipWithT @2 (applyForn n) rest (map2Sh n angles x))
  where (angles,rest) = splitParams'  n prms
  

forn :: forall n t. (KnownPeano n, KnownFloat t) => 
        RnnCell t '[ '[Pow2 n] ] (Tensor '[OrnEmbSize n] t) (Tensor '[ Pow2 n ] t)
forn xt = C $ \(VecSing ht1) ->
                knownPowTwo @n ?>
                knownTwo (knownPeano @n) ?>
                knownOrnAnglesShape (knownPeano @n) ?>
                knownProduct @(OrnAnglesShape n) ?>
                let ht = applyForn (knownPeano @n) (reshape xt)  (reshape ht1)
                in (VecSing (reshape ht) , reshape ht)

mkFORN :: ∀ n. (KnownPeano n) => DropProb -> Gen (RnnCell Float32 '[ '[Pow2 n] ] (Tensor '[OrnEmbSize n] Float32) (Tensor '[Pow2 n] Float32))
mkFORN dropProb = do
  rdrp1 <- knownPowTwo @n ?> mkDropout dropProb
  return (onStates (\(VecSing r) -> VecSing (rdrp1 r)) (forn @n) )



