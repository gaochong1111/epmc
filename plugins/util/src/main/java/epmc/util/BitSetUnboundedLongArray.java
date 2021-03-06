/****************************************************************************

    ePMC - an extensible probabilistic model checker
    Copyright (C) 2017

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

 *****************************************************************************/

package epmc.util;

import java.util.Arrays;

public final class BitSetUnboundedLongArray implements BitSet {
    private final static int WORD_SIZE = Long.SIZE;
    private static final int LOG2LONGSIZE = 6;
    private static final long ONES = 0xffffffffffffffffL;
    private long[] content;

    public BitSetUnboundedLongArray(int size) {
        assert size >= 0;
        int numEntries = size / WORD_SIZE + (size % WORD_SIZE != 0 ? 1 : 0);
        numEntries = numEntries == 0 ? 1 : numEntries;
        content = new long[numEntries];
    }

    public BitSetUnboundedLongArray(long[] content) {
        assert content != null;
        this.content = content.clone();
    }

    public BitSetUnboundedLongArray() {
        this(WORD_SIZE);
    }

    @Override
    public void set(int bitIndex, boolean value) {
        assert bitIndex >= 0 : bitIndex;
        int offset = offset(bitIndex);
        ensureSize(offset);
        if (value) {
            content[offset] |= bit(bitIndex);
        } else {
            content[offset] &= ~bit(bitIndex);
        }
    }

    private void ensureSize(int offset) {
        assert offset >= 0;
        int size = content.length;
        if (offset < size) {
            return;
        }
        int newSize = size;
        while (newSize <= offset) {
            newSize *= 2;
        }
        content = Arrays.copyOf(content, newSize);
    }

    @Override
    public boolean get(int bitIndex) {
        assert bitIndex >= 0;
        int size = content.length * WORD_SIZE;
        if (bitIndex >= size) {
            return false;
        }
        int offset = offset(bitIndex);
        return (content[offset] & bit(bitIndex)) != 0L;
    }

    @Override
    public int size() {
        return content.length * WORD_SIZE;
    }

    @Override
    public BitSet clone() {
        return new BitSetUnboundedLongArray(content);
    }

    @Override
    public boolean equals(Object obj) {
        assert obj != null;
        if (obj instanceof BitSetUnboundedLongArray) {
            BitSetUnboundedLongArray other = (BitSetUnboundedLongArray) obj;
            int min = Math.min(content.length, other.content.length);
            for (int i = 0; i < min; i++) {
                if (content[i] != other.content[i]) {
                    return false;
                }
            }
            for (int i = min; i < content.length; i++) {
                if (content[i] != 0L) {
                    return false;
                }
            }
            for (int i = min; i < other.content.length; i++) {
                if (other.content[i] != 0L) {
                    return false;
                }
            }
            return true;
        } else {
            return UtilBitSet.equals(this, obj);
        }
    }

    @Override
    public int hashCode() {
        int end;
        for (end = content.length - 1; end >= 0; end--) {
            if (content[end] != 0L) {
                break;
            }
        }
        end++;
        int hash = 0;
        for (int word = 0; word < end; word++) {
            hash = Long.hashCode(content[word]) + (hash << 6) + (hash << 16) - hash;
        }
        return hash;
    }

    @Override
    public void clear(int bitIndex) {
        assert bitIndex >= 0 : bitIndex;
        int offset = offset(bitIndex);
        ensureSize(offset);
        content[offset] &= ~bit(bitIndex);
    }

    @Override
    public void set(int bitIndex) {
        assert bitIndex >= 0 : bitIndex;
        int offset = offset(bitIndex);
        ensureSize(offset);
        content[offset] |= bit(bitIndex);
    }

    @Override
    public void clear() {
        Arrays.fill(content, 0L);
    }

    @Override
    public void or(BitSet other) {
        assert other != null;
        if (other instanceof BitSetUnboundedLongArray) {
            BitSetUnboundedLongArray otherUB = (BitSetUnboundedLongArray) other;
            int to;
            for (to = otherUB.content.length - 1; to >= 0; to--) {
                if (otherUB.content[to] != 0L) {
                    break;
                }
            }
            to++;
            if (to > content.length) {
                ensureSize(to);
            }
            for (int i = 0; i < to; i++) {
                content[i] = content[i] | otherUB.content[i];
            }
        } else {
            BitSet.super.or(other);
        }
    }

    @Override
    public void andNot(BitSet other) {
        assert other != null;
        if (other instanceof BitSetUnboundedLongArray) {
            BitSetUnboundedLongArray otherUB = (BitSetUnboundedLongArray) other;
            int min = Math.min(this.content.length, otherUB.content.length);
            for (int i = 0; i < min; i++) {
                content[i] = content[i] & ~otherUB.content[i];
            }
        } else {
            BitSet.super.andNot(other);
        }
    }

    @Override
    public void and(BitSet other) {
        assert other != null;
        if (other instanceof BitSetUnboundedLongArray) {
            BitSetUnboundedLongArray otherUB = (BitSetUnboundedLongArray) other;
            int min = Math.min(this.content.length, otherUB.content.length);
            for (int i = 0; i < min; i++) {
                content[i] = content[i] & otherUB.content[i];
            }
            for (int i = min; i < content.length; i++) {
                content[i] = 0L;
            }
        } else {
            BitSet.super.and(other);
        }
    }

    @Override
    public boolean isEmpty() {
        for (int i = 0; i < content.length; i++) {
            if (content[i] != 0L) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int cardinality() {
        int result = 0;
        for (int i = 0; i < content.length; i++) {
            result += bitCount(content[i]);
        }
        return result;
    }

    /**
     * See
     * "Algorithms and data structures with applications to 
     *  graphics and geometry", by Jurg Nievergelt and Klaus Hinrichs,
     *  Prentice Hall, 1993.
     * @param value
     * @return
     */
    private static int bitCount(long value) {
        value -= (value & 0xaaaaaaaaaaaaaaaaL) >>> 1;
        value =  (value & 0x3333333333333333L) + ((value >>> 2) & 0x3333333333333333L);
        value =  (value + (value >>> 4)) & 0x0f0f0f0f0f0f0f0fL;
        value += value >>> 8;
        value += value >>> 16;
        return ((int)(value) + (int)(value >>> 32)) & 0xff;
    }

    @Override
    public void flip(int bitIndex) {
        assert bitIndex >= 0 : bitIndex;
        int offset = offset(bitIndex);
        ensureSize(offset);
        content[offset] ^= bit(bitIndex);
    }

    @Override
    public boolean intersects(BitSet set) {
        assert set != null;
        if (set instanceof BitSetUnboundedLongArray) {
            BitSetUnboundedLongArray other = (BitSetUnboundedLongArray) set;
            int min = Math.min(content.length, other.content.length);
            for (int i = 0; i < min; i++) {
                if ((content[i] & other.content[i]) != 0L) {
                    return true;
                }
            }
            return false;
        } else {
            return BitSet.super.intersects(set);
        }
    }

    @Override
    public void xor(BitSet other) {
        assert other != null;
        if (other instanceof BitSetUnboundedLongArray) {
            BitSetUnboundedLongArray otherUB = (BitSetUnboundedLongArray) other;
            int to;
            for (to = otherUB.content.length - 1; to >= 0; to--) {
                if (otherUB.content[to] != 0L) {
                    break;
                }
            }
            to++;
            if (to > content.length) {
                ensureSize(to);
            }
            for (int i = 0; i < to; i++) {
                content[i] = content[i] ^ otherUB.content[i];
            }
        } else {
            BitSet.super.or(other);
        }
    }

    @Override
    public int nextSetBit(int index) {
        assert index >= 0;
        int offset = offset(index);
        if (offset >= content.length) {
            return -1;
        }
        long word = content[offset] & (ONES << index);
        while (true) {
            if (word != 0L) {
                return (offset * WORD_SIZE) + Long.numberOfTrailingZeros(word);
            }
            offset++;
            if (offset >= content.length) {
                return -1;
            }
            word = content[offset];
        }
    }

    @Override
    public int nextClearBit(int index) {
        assert index >= 0;
        int offset = offset(index);
        if (offset >= content.length) {
            return index;
        }
        long word = ~content[offset] & (ONES << index);
        while (true) {
            if (word != 0) {
                return (offset * WORD_SIZE) + Long.numberOfTrailingZeros(word);
            }
            offset++;
            if (offset >= content.length) {
                return offset * WORD_SIZE;
            }
            word = ~content[offset];
        }
    }

    @Override
    public int length() {
        for (int offset = content.length - 1; offset >= 0; offset--) {
            long word = content[offset];
            if (word != 0L) {
                return offset * WORD_SIZE + WORD_SIZE - Long.numberOfLeadingZeros(word);
            }
        }
        return 0;
    }

    @Override
    public void clear(int fromIndex, int toIndex) {
        assert fromIndex >= 0;
        assert fromIndex <= toIndex;
        if (fromIndex == toIndex) {
            return;
        }
        int fromOffset = offset(fromIndex);
        if (fromOffset >= content.length) {
            return;
        }
        int toOffset = offset(toIndex - 1);
        if (toOffset >= content.length) {
            toOffset = content.length - 1;
        }
        long fromOffsetMask = ONES << fromIndex;
        long toOffsetMask  = ONES >>> -toIndex;
        if (fromOffset == toOffset) {
            content[fromOffset] &= ~(fromOffsetMask & toOffsetMask);
        } else {
            content[fromOffset] &= ~fromOffsetMask;
            for (int offset = fromOffset+1; offset < toOffset; offset++) {
                content[offset] = 0;
            }
            content[toOffset] &= ~toOffsetMask;            
        }
    }

    @Override
    public void set(int fromIndex, int toIndex) {
        assert fromIndex >= 0;
        assert fromIndex <= toIndex;
        if (fromIndex == toIndex) {
            return;
        }
        int fromOffset = offset(fromIndex);
        int toOffset = offset(toIndex - 1);
        ensureSize(toOffset);
        long fromOffsetMask = ONES << fromIndex;
        long toOffsetMask  = ONES >>> -toIndex;
        if (fromOffset == toOffset) {
            content[fromOffset] |= (fromOffsetMask & toOffsetMask);
        } else {
            content[fromOffset] |= fromOffsetMask;
            for (int offset = fromOffset+1; offset < toOffset; offset++) {
                content[offset] = ONES;
            }
            content[toOffset] |= toOffsetMask;
        }
    }

    @Override
    public void flip(int fromIndex, int toIndex) {
        assert fromIndex >= 0;
        assert fromIndex <= toIndex;
        if (fromIndex == toIndex) {
            return;
        }
        int fromOffset = offset(fromIndex);
        int toOffset = offset(toIndex - 1);
        ensureSize(toOffset);
        long fromOffsetMask = ONES << fromIndex;
        long toOffsetMask  = ONES >>> -toIndex;
        if (fromOffset == toOffset) {
            content[fromOffset] ^= (fromOffsetMask & toOffsetMask);
        } else {
            content[fromOffset] ^= fromOffsetMask;
            for (int offset = fromOffset+1; offset < toOffset; offset++) {
                content[offset] ^= ONES;
            }
            content[toOffset] ^= toOffsetMask;
        }
    }

    @Override
    public void set(int fromIndex, int toIndex, boolean value) {
        if (value) {
            set(fromIndex, toIndex);
        } else {
            clear(fromIndex, toIndex);
        }
    }

    private long bit(int index) {
        assert index >= 0;
        return 1L << index;
    }

    private int offset(int index) {
        assert index >= 0;
        return index >> LOG2LONGSIZE;        
    }

    @Override
    public void and(BitSet operand1, BitSet operand2) {
        assert operand1 != null;
        assert operand2 != null;
        if (!(operand1 instanceof BitSetUnboundedLongArray)
                || !(operand2 instanceof BitSetUnboundedLongArray)) {
            BitSet.super.and(operand1, operand2);
            return;
        }
        BitSetUnboundedLongArray other1 = (BitSetUnboundedLongArray) operand1;
        BitSetUnboundedLongArray other2 = (BitSetUnboundedLongArray) operand2;
        int otherlen;
        for (otherlen = other1.content.length - 1; otherlen >= 0; otherlen--) {
            if (other1.content[otherlen] != 0L) {
                break;
            }
        }
        for (; otherlen >= 0; otherlen--) {
            if (other2.content[otherlen] != 0L) {
                break;
            }
        }
        otherlen++;
        for (int offset = otherlen; offset < content.length; offset++) {
            content[offset] = 0L;
        }
        ensureSize(otherlen - 1);
        for (int i = 0; i < otherlen; i++) {
            content[i] = other1.content[i] & other2.content[i];
        }
    }

    @Override
    public void andNot(BitSet operand1, BitSet operand2) {
        assert operand1 != null;
        assert operand2 != null;
        if (!(operand1 instanceof BitSetUnboundedLongArray)
                || !(operand2 instanceof BitSetUnboundedLongArray)) {
            BitSet.super.and(operand1, operand2);
            return;
        }
        BitSetUnboundedLongArray other1 = (BitSetUnboundedLongArray) operand1;
        BitSetUnboundedLongArray other2 = (BitSetUnboundedLongArray) operand2;
        int otherlen;
        for (otherlen = other1.content.length - 1; otherlen >= 0; otherlen--) {
            if (other1.content[otherlen] != 0L) {
                break;
            }
        }
        for (; otherlen >= 0; otherlen--) {
            if (other2.content[otherlen] != 0L) {
                break;
            }
        }
        otherlen++;
        for (int offset = otherlen; offset < content.length; offset++) {
            content[offset] = 0L;
        }
        ensureSize(otherlen - 1);
        for (int i = 0; i < otherlen; i++) {
            content[i] = other1.content[i] & ~other2.content[i];
        }
    }

    @Override
    public void or(BitSet operand1, BitSet operand2) {
        assert operand1 != null;
        assert operand2 != null;
        if (!(operand1 instanceof BitSetUnboundedLongArray)
                || !(operand2 instanceof BitSetUnboundedLongArray)) {
            BitSet.super.or(operand1, operand2);
            return;
        }
        BitSetUnboundedLongArray other1 = (BitSetUnboundedLongArray) operand1;
        BitSetUnboundedLongArray other2 = (BitSetUnboundedLongArray) operand2;
        int other1len;
        for (other1len = other1.content.length - 1; other1len >= 0; other1len--) {
            if (other1.content[other1len] != 0L) {
                break;
            }
        }
        other1len++;
        int other2len;
        for (other2len = other2.content.length - 1; other2len >= 0; other2len--) {
            if (other2.content[other2len] != 0L) {
                break;
            }
        }
        other2len++;
        int otherlen = Math.max(other1len, other2len);
        for (int offset = otherlen; offset < content.length; offset++) {
            content[offset] = 0L;
        }
        ensureSize(otherlen - 1);
        for (int offset = 0; offset < otherlen; offset++) {
            content[offset] = other1.content[offset] | other2.content[offset];
        }
    }

    @Override
    public void xor(BitSet operand1, BitSet operand2) {
        assert operand1 != null;
        assert operand2 != null;
        if (!(operand1 instanceof BitSetUnboundedLongArray)
                || !(operand2 instanceof BitSetUnboundedLongArray)) {
            BitSet.super.and(operand1, operand2);
            return;
        }
        BitSetUnboundedLongArray other1 = (BitSetUnboundedLongArray) operand1;
        BitSetUnboundedLongArray other2 = (BitSetUnboundedLongArray) operand2;
        int other1len;
        for (other1len = other1.content.length - 1; other1len >= 0; other1len--) {
            if (other1.content[other1len] != 0L) {
                break;
            }
        }
        other1len++;
        int other2len;
        for (other2len = other2.content.length - 1; other2len >= 0; other2len--) {
            if (other2.content[other2len] != 0L) {
                break;
            }
        }
        other2len++;
        int otherlen = Math.max(other1len, other2len);
        for (int offset = otherlen; offset < content.length; offset++) {
            content[offset] = 0L;
        }
        ensureSize(otherlen - 1);
        for (int offset = 0; offset < otherlen; offset++) {
            content[offset] = other1.content[offset] ^ other2.content[offset];
        }
    }

    @Override
    public String toString() {
        return UtilBitSet.toString(this);
    }
}
