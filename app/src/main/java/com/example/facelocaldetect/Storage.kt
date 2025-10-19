package com.example.facelocaldetect

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

data class Person(val name: String, val vectors: List<FloatArray>)

class LocalDB(private val context: Context) {
    private val file: File = File(context.filesDir, "face_vectors.json")

    fun saveAll(persons: List<Person>) {
        val arr = JSONArray()
        persons.forEach { p ->
            val obj = JSONObject()
            obj.put("name", p.name)
            val vArr = JSONArray()
            p.vectors.forEach { vec ->
                val one = JSONArray()
                vec.forEach { one.put(it) }
                vArr.put(one)
            }
            obj.put("vectors", vArr)
            arr.put(obj)
        }
        file.writeText(arr.toString())
    }

    fun loadAll(): List<Person> {
        if (!file.exists()) return emptyList()
        val txt = file.readText()
        val arr = JSONArray(txt)
        val res = mutableListOf<Person>()

        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            val name = obj.getString("name")
            val vArr = obj.getJSONArray("vectors")
            val list = mutableListOf<FloatArray>()
            for (j in 0 until vArr.length()) {
                val one = vArr.getJSONArray(j)
                val vec = FloatArray(one.length())
                for (k in 0 until one.length()) {
                    vec[k] = one.getDouble(k).toFloat()
                }
                list.add(vec)
            }
            res.add(Person(name, list))
        }
        return res
    }

    fun clear() {
        if (file.exists()) file.delete()
    }
}

object MathUtil {
    fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        var na = 0f
        var nb = 0f
        for (i in a.indices) {
            dot += a[i]*b[i]
            na += a[i]*a[i]
            nb += b[i]*b[i]
        }
        if (na == 0f || nb == 0f) return 0f
        return dot / (kotlin.math.sqrt(na) * kotlin.math.sqrt(nb))
    }
}